from transformers import (
    pipeline, AutoTokenizer, TrainingArguments, Trainer, AutoModelForSeq2SeqLM
)
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


class LLMStockPredictBase():
    def __init__(
            self, model_name, r=8, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05,
            bias="none", task_type=TaskType.SEQ_2_SEQ_LM
        ):
        self.model_name: str = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=task_type,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model = get_peft_model(model, self.lora_config)
        self.pipe = pipeline(
            "text2text-generation", model=self.model, tokenizer=self.tokenizer, max_new_tokens=512
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        self.chain = None

    def train(
            self, train_dataset, val_dataset, save=True, save_dir=None, per_device_train_batch_size=2,
            per_device_eval_batch_size=2, num_train_epochs=1, logging_dir="./logs", save_strategy="no",
            eval_strategy="no", label_names=["labels"], fp16=True, load_best_model_at_end=True,
            report_to="none"
        ):
        if save_dir is None:
            save_dir = "./" + self.model_name.split('/')[-1]

        # training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            logging_dir=logging_dir,
            save_strategy=save_strategy,
            eval_strategy=eval_strategy,
            output_dir=save_dir + '/lora-checkpoints',
            label_names=label_names,
            fp16=fp16,
            load_best_model_at_end=load_best_model_at_end,
            report_to=report_to
        )

        # trainer setup
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )
        trainer.train()

        # save adapter
        if save:
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
        # fine-tuned pipeline
        self.llm = HuggingFacePipeline(pipeline=pipeline(
            "text2text-generation", model=self.model, tokenizer=self.tokenizer, max_new_tokens=512
        ))

    def prompt_template_chain(self, template: str, input_variables: list[str]):
        prompt = PromptTemplate(input_variables=input_variables, template=template)
        self.chain = LLMChain(llm=self.llm, prompt=prompt)

    def invoke(self, input: dict[str, any]):
        if self.chain is None:
            return None
        return self.chain.invoke(input)

    def load(self, load_dir: str):
        base_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        model = PeftModel.from_pretrained(base_model, load_dir)
        self.model = model.merge_and_unload()
