import os
import json
import torch
import torch.nn as nn
from transformers import BertModel, BertForSequenceClassification
from sklearn.metrics import classification_report


class HBERTPredictBase(nn.Module):
    def __init__(
        self, model, loss_fn, num_labels=3, max_chunk_length=256, aggregation_method='attention',
        num_stocks=10, stock_embedding_dim=16, output_hidden_states=False
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.hidden_size = self.model.config.hidden_size

        self.pooler = nn.Linear(self.hidden_size, self.hidden_size)
        self.pooler_activation = nn.Tanh()
        self.aggregation_method = aggregation_method
        if aggregation_method == 'attention':
            self.attention = nn.Linear(self.hidden_size, 1)

        self.temporal_embedding = nn.Linear(1, 32)
        self.temporal_activation = nn.Tanh()
        self.stock_embedding = nn.Embedding(num_stocks, stock_embedding_dim)
        combined_dim = self.hidden_size + 32 + stock_embedding_dim

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )
        self.max_chunk_length = max_chunk_length
        self.output_hidden_states = output_hidden_states

    def forward(self, input_ids_list, attention_mask_list, time_values, stock_indices):
        """
        Forward pass with consideration to time and stock. Uses a variety of techniques such
        as dropout.
        """
        batch_size = len(input_ids_list)
        chunk_embeddings = []

        for i in range(batch_size):
            example_chunks = []
            for j in range(len(input_ids_list[i])):
                # get the output of last hidden state
                if not self.output_hidden_states:
                    out = self.model(
                        input_ids=input_ids_list[i][j].unsqueeze(0),
                        attention_mask=attention_mask_list[i][j].unsqueeze(0)
                    )
                    cls = out.last_hidden_state[:, 0, :]
                else:
                    out = self.model(
                        input_ids=input_ids_list[i][j].unsqueeze(0),
                        attention_mask=attention_mask_list[i][j].unsqueeze(0),
                        output_hidden_states=True
                    )
                    cls = out.hidden_states[-1][:, 0, :]
                # pooled and append to example chunks
                pooled = self.pooler(cls)
                pooled = self.pooler_activation(pooled)
                example_chunks.append(pooled)

            stacked = torch.cat(example_chunks, dim=0)
            if self.aggregation_method == 'mean':
                aggregated = torch.mean(stacked, dim=0, keepdim=True)
            elif self.aggregation_method == 'max':
                aggregated = torch.max(stacked, dim=0, keepdim=True)[0]
            elif self.aggregation_method == 'attention':
                attn_weights = torch.softmax(self.attention(stacked), dim=0)
                aggregated = torch.sum(stacked * attn_weights, dim=0, keepdim=True)
            chunk_embeddings.append(aggregated)

        text_embeddings = torch.cat(chunk_embeddings, dim=0)
        time_embeddings = self.temporal_activation(self.temporal_embedding(time_values.unsqueeze(1)))
        stock_embeddings = self.stock_embedding(stock_indices)

        combined = torch.cat([text_embeddings, time_embeddings, stock_embeddings], dim=1)
        combined = self.dropout(combined)
        return self.classifier(combined)
    
    def train_model(
            self, train_loader, test_loader, device, num_epochs=45, learning_rate=2e-5, focal_gamma=2.0
        ):
        """
        Model training call.
        """
        self.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        print("Model and data prepared. Ready to train!")
        loss_fn = self.loss_fn(gamma=focal_gamma, reduction='mean')

        for epoch in range(num_epochs):
            self.train()
            total_loss, correct, total = 0, 0, 0
            for batch in train_loader:
                input_ids_list, attention_mask_list, time_values, stock_indices, labels = batch
                input_ids_list = [x.to(device) for x in input_ids_list]
                attention_mask_list = [x.to(device) for x in attention_mask_list]
                time_values = time_values.to(device)
                stock_indices = stock_indices.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = self(input_ids_list, attention_mask_list, time_values, stock_indices)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()

            acc = 100. * correct / total
            print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={acc:.2f}%")

        print("\nTraining complete. Evaluating on test set...")
        self.eval()
        test_preds, test_labels_list = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids_list, attention_mask_list, time_values, stock_indices, labels = batch
                input_ids_list = [x.to(device) for x in input_ids_list]
                attention_mask_list = [x.to(device) for x in attention_mask_list]
                time_values = time_values.to(device)
                stock_indices = stock_indices.to(device)
                labels = labels.to(device)

                outputs = self(input_ids_list, attention_mask_list, time_values, stock_indices)
                _, preds = outputs.max(1)
                test_preds.extend(preds.cpu().numpy())
                test_labels_list.extend(labels.cpu().numpy())

        print("Classification Report:")
        print(classification_report(test_labels_list, test_preds, target_names=['Decrease', 'Increase']))

    def save_model(self, model_config, save_dir, tokenizer):
        """
        Save the model, tokenizer, and associated metadata

        Args:
            tokenizer: BERT tokenizer used with the model
            save_dir : directory path where the model should be saved
        """
        os.makedirs(save_dir, exist_ok=True)

        # save model state and tokenizer
        torch.save(self.model.state_dict(), f"{save_dir}/hbert_model.pt")
        tokenizer.save_pretrained(save_dir)
        with open(f"{save_dir}/model_config.json", 'w') as f:
            json.dump(model_config, f)

        print(f"Model successfully saved to {save_dir}")
        return save_dir

    @staticmethod
    def load_model(save_dir, bert_model, loss_fn, device):
        """
        Load a saved model, tokenizer and its metadata.

        Args:
            save_dir: directory path containing the saved model

        Returns:
            A tuple containing (model, tokenizer)
        """
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"Model directory {save_dir} not found")

        # load model configuration
        with open(f"{save_dir}/model_config.json", 'r') as f:
            model_config = json.load(f)

        # initialize model with saved configuration
        model = HBERTPredictBase(bert_model, loss_fn, **model_config)

        # load model weights
        model.model.load_state_dict(torch.load(f"{save_dir}/hbert_model.pt", map_location=device))
        model.to(device)
        model.eval()
        print(f"Model successfully loaded from {save_dir}")
        return model


class HierarchicalBERTStockPredict(nn.Module):
    """
    Hierarchical BERT for Stock Prediction.
    """
    def __init__(
        self, num_labels=3, max_chunk_length=256, aggregation_method='attention',
        num_stocks=10, stock_embedding_dim=16
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_hidden_size = self.bert.config.hidden_size

        self.bert_pooler = nn.Linear(self.bert_hidden_size, self.bert_hidden_size)
        self.bert_pooler_activation = nn.Tanh()
        self.aggregation_method = aggregation_method
        if aggregation_method == 'attention':
            self.attention = nn.Linear(self.bert_hidden_size, 1)

        self.temporal_embedding = nn.Linear(1, 32)
        self.temporal_activation = nn.Tanh()
        self.stock_embedding = nn.Embedding(num_stocks, stock_embedding_dim)
        combined_dim = self.bert_hidden_size + 32 + stock_embedding_dim

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )
        self.max_chunk_length = max_chunk_length

    def forward(self, input_ids_list, attention_mask_list, time_values, stock_indices):
        """
        Forward pass.
        """
        batch_size = len(input_ids_list)
        chunk_embeddings = []

        for i in range(batch_size):
            example_chunks = []
            for j in range(len(input_ids_list[i])):
                out = self.bert(
                    input_ids=input_ids_list[i][j].unsqueeze(0),
                    attention_mask=attention_mask_list[i][j].unsqueeze(0)
                )
                cls = out.last_hidden_state[:, 0, :]
                pooled = self.bert_pooler(cls)
                pooled = self.bert_pooler_activation(pooled)
                example_chunks.append(pooled)

            stacked = torch.cat(example_chunks, dim=0)
            if self.aggregation_method == 'mean':
                aggregated = torch.mean(stacked, dim=0, keepdim=True)
            elif self.aggregation_method == 'max':
                aggregated = torch.max(stacked, dim=0, keepdim=True)[0]
            elif self.aggregation_method == 'attention':
                attn_weights = torch.softmax(self.attention(stacked), dim=0)
                aggregated = torch.sum(stacked * attn_weights, dim=0, keepdim=True)
            chunk_embeddings.append(aggregated)

        text_embeddings = torch.cat(chunk_embeddings, dim=0)
        time_embeddings = self.temporal_activation(self.temporal_embedding(time_values.unsqueeze(1)))
        stock_embeddings = self.stock_embedding(stock_indices)

        combined = torch.cat([text_embeddings, time_embeddings, stock_embeddings], dim=1)
        combined = self.dropout(combined)
        return self.classifier(combined)


class HierarchicalFINBERTStockPredict(nn.Module):
    """
    Hierarchical FINBERT for Stock Prediction.
    """
    def __init__(
        self, num_labels=3, max_chunk_length=256, aggregation_method='attention',
        num_stocks=10, stock_embedding_dim=16
    ):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            'yiyanghkust/finbert-tone', num_labels=num_labels, ignore_mismatched_sizes=True
        )
        self.hidden_size = self.model.config.hidden_size

        self.pooler = nn.Linear(self.hidden_size, self.hidden_size)
        self.pooler_activation = nn.Tanh()
        self.aggregation_method = aggregation_method
        if aggregation_method == 'attention':
            self.attention = nn.Linear(self.hidden_size, 1)

        self.temporal_embedding = nn.Linear(1, 32)
        self.temporal_activation = nn.Tanh()
        self.stock_embedding = nn.Embedding(num_stocks, stock_embedding_dim)
        combined_dim = self.hidden_size + 32 + stock_embedding_dim

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )
        self.max_chunk_length = max_chunk_length

    def forward(self, input_ids_list, attention_mask_list, time_values, stock_indices):
        """
        Forward pass with consideration to time and stock. Uses a variety of techniques such
        as dropout.
        """
        batch_size = len(input_ids_list)
        chunk_embeddings = []

        for i in range(batch_size):
            example_chunks = []
            for j in range(len(input_ids_list[i])):
                # get the output of last hidden state
                out = self.model(
                    input_ids=input_ids_list[i][j].unsqueeze(0),
                    attention_mask=attention_mask_list[i][j].unsqueeze(0),
                    output_hidden_states=True
                )
                cls = out.hidden_states[-1][:, 0, :]
                # pooled and append to example chunks
                pooled = self.pooler(cls)
                pooled = self.pooler_activation(pooled)
                example_chunks.append(pooled)

            stacked = torch.cat(example_chunks, dim=0)
            if self.aggregation_method == 'mean':
                aggregated = torch.mean(stacked, dim=0, keepdim=True)
            elif self.aggregation_method == 'max':
                aggregated = torch.max(stacked, dim=0, keepdim=True)[0]
            elif self.aggregation_method == 'attention':
                attn_weights = torch.softmax(self.attention(stacked), dim=0)
                aggregated = torch.sum(stacked * attn_weights, dim=0, keepdim=True)
            chunk_embeddings.append(aggregated)

        text_embeddings = torch.cat(chunk_embeddings, dim=0)
        time_embeddings = self.temporal_activation(self.temporal_embedding(time_values.unsqueeze(1)))
        stock_embeddings = self.stock_embedding(stock_indices)

        combined = torch.cat([text_embeddings, time_embeddings, stock_embeddings], dim=1)
        combined = self.dropout(combined)
        return self.classifier(combined)


def train_hbert(
        model, train_loader, test_loader, device, loss_fn,
        num_epochs=45, learning_rate=2e-5, focal_gamma=2.0
    ):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print("Model and data prepared. Ready to train!")
    loss_fn = loss_fn(gamma=focal_gamma, reduction='mean')

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in train_loader:
            input_ids_list, attention_mask_list, time_values, stock_indices, labels = batch
            input_ids_list = [x.to(device) for x in input_ids_list]
            attention_mask_list = [x.to(device) for x in attention_mask_list]
            time_values = time_values.to(device)
            stock_indices = stock_indices.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids_list, attention_mask_list, time_values, stock_indices)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        acc = 100. * correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={acc:.2f}%")

    print("\nTraining complete. Evaluating on test set...")
    model.eval()
    test_preds, test_labels_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids_list, attention_mask_list, time_values, stock_indices, labels = batch
            input_ids_list = [x.to(device) for x in input_ids_list]
            attention_mask_list = [x.to(device) for x in attention_mask_list]
            time_values = time_values.to(device)
            stock_indices = stock_indices.to(device)
            labels = labels.to(device)

            outputs = model(input_ids_list, attention_mask_list, time_values, stock_indices)
            _, preds = outputs.max(1)
            test_preds.extend(preds.cpu().numpy())
            test_labels_list.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(test_labels_list, test_preds, target_names=['Decrease', 'Increase']))
