from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast
from transformers import AutoConfig
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2LMHeadModel
from transformers import TrainingArguments, Trainer
import datasets

####################################################################
# Step 1 -- Get raw dataset

train_data, test_data = datasets.load_dataset(
        #'openwebtext',
        'wikitext', 'wikitext-103-v1',
        split =['train[:70%]', 'test[:30%]']
    )

data_splits = datasets.DatasetDict({'train': train_data, 
                           'validation': test_data})


####################################################################
# Step 2 -- set up tokenizer

use_default_tokenizer = True
MAX_LENGTH = 512

if use_default_tokenizer == True:
    # Step 2A -- Use existing GPT-2 tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', pad_token='<|endoftext|>')
    tokenizer.model_max_length = MAX_LENGTH
else:
    # Step 2B -- Train custom GPT-2 tokenizer 
    VOCAB_DIR = "vocab"
    vocab_files = [str(x) for x in Path(VOCAB_DIR).glob('*.txt')]

    default_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    vocab_size = default_tokenizer.vocab_size
    #model_max_length = default_tokenizer.model_max_length

    # Train custom tokenizer
    custom_tokenizer = ByteLevelBPETokenizer(lowercase=True)
    custom_tokenizer.train(
                files = vocab_files, 
                vocab_size = vocab_size, 
                min_frequency = 1, 
                special_tokens=['<|endoftext|>'])
    custom_tokenizer.enable_truncation(max_length = MAX_LENGTH)
    tokenizer = custom_tokenizer

####################################################################
# Step 3 -- Create tokenized dataset

def tokenize(element):
    outputs = tokenizer(element['text'], 
                        truncation=True, 
                        max_length=MAX_LENGTH, 
                        return_overflowing_tokens=True, 
                        return_length=True)
    input_batch = []
    for length, input_ids in zip(outputs['length'], outputs['input_ids']):
        if length == MAX_LENGTH:
            input_batch.append(input_ids)
    return {'input_ids': input_batch}

num_proc = 1
tokenized_datasets = data_splits.map(
                        tokenize, 
                        batched = True, 
                        num_proc = num_proc,
                        remove_columns = train_data.column_names
                    )


####################################################################
# Step 4 -- Pre-train GPT-2 model

config = AutoConfig.from_pretrained(
                        'gpt2', 
                        vocab_size = tokenizer.vocab_size, 
                        n_ctx = MAX_LENGTH, 
                        bos_token_id = tokenizer.bos_token_id, 
                        eos_token_id = tokenizer.eos_token_id
                    )

# Create the DataCollator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)

TRAIN_EPOCHS = 1
SAVE_STEPS = 1000
SAVE_TOTAL_LIMIT = 2

args = TrainingArguments(output_dir='tmp/checkpoints', 
                             overwrite_output_dir=True, 
                             optim='adamw_torch',
                             per_device_train_batch_size = 1, 
                             evaluation_strategy = 'epoch', 
                             num_train_epochs = TRAIN_EPOCHS, 
                             weight_decay = 0.1, 
                             warmup_steps = 1_000, 
                             lr_scheduler_type = 'cosine', 
                             learning_rate = 5e-4,
                             save_steps = SAVE_STEPS, 
                             save_total_limit = SAVE_TOTAL_LIMIT)

clm = GPT2LMHeadModel(config)

trainer = Trainer(
            model = clm, 
            tokenizer = tokenizer, 
            args = args, 
            data_collator = data_collator, 
            train_dataset = tokenized_datasets['train'],
            eval_dataset = tokenized_datasets['validation']
        )

trainer.train()