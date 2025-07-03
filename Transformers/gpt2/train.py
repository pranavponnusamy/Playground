from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import os
from torch.utils.data import DataLoader
from model import *
from dataclasses import dataclass
import wandb
import torch
import torch.nn as nn

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

@dataclass
class trainConfig:
    epochs: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 16
    max_length: int = 128
    log_interval: int = 10
    save_interval: int = 1

def create_trainloader(tc: trainConfig):
    d_train = load_dataset("roneneldan/TinyStories", split="train")
    d_val = load_dataset("roneneldan/TinyStories", split="validation")
    tokenizer = AutoTokenizer.from_pretrained('GPT2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def tokenization(example):
        return tokenizer(example["text"], truncation=True, max_length=tc.max_length)

    # d_train = d_train.map(tokenization, batched=True, remove_columns=ds.column_names).select(range(100))
    d_train = d_train.map(tokenization, batched=True, remove_columns=d_train.column_names).select(range(1000))
    d_val = d_val.map(tokenization, batched=True, remove_columns=d_val.column_names).select(range(1000))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt", padding=True)

    trainloader = DataLoader(d_train, batch_size=tc.batch_size, collate_fn=data_collator, shuffle=True)
    testloader = DataLoader(d_val, batch_size=tc.batch_size, collate_fn=data_collator, shuffle=True)
    
    return trainloader, testloader

def validate_model(gpt: GPT, testloader: DataLoader, criterion):
    gpt.eval()
    total_val_loss = 0
    val_steps = 0
    
    with torch.no_grad():
        for batch in testloader:
            input_ids = batch['input_ids'].to(config.device)
            padding_mask = batch['attention_mask'].to(config.device)
            targets = input_ids[:, 1:].contiguous()  # Shift targets by 1
            
            #Need to remove the last token from inputs and padding_mask
            inputs = input_ids[:, :-1].contiguous()  
            padding_mask = padding_mask[:, :-1].contiguous() 
            
            outputs = gpt(inputs, padding_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            total_val_loss += loss.item()
            val_steps += 1
            
            del input_ids
            del padding_mask
    
    avg_val_loss = total_val_loss / val_steps
    gpt.train()
    return avg_val_loss

def train_model(gpt: GPT, trainloader: DataLoader, testloader: DataLoader, tc: trainConfig):
    wandb.config.update({
        "epochs": tc.epochs,
        "learning_rate": tc.learning_rate,
        "batch_size": tc.batch_size,
        "max_length": tc.max_length,
        "model_config": {
            "vocab_size": config.vocab_size,
            "n_layers": config.num_attention_blocks,
            "n_heads": config.num_attention_heads,
            "d_model": config.embedding_dim,
            "d_ff": config.ff_hidden_dim,
            "max_seq_len": config.max_seq_len,
            "dropout": config.dropout
        }
    })
    
    wandb.watch(gpt, log="all", log_freq=100)
    
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=tc.learning_rate)
    criterion = nn.CrossEntropyLoss()
    gpt.train()
    
    global_step = 0
    
    for epoch in range(tc.epochs): 
        total_loss = 0
        epoch_steps = 0
        
        for batch_idx, batch in enumerate(trainloader):
            input_ids = batch['input_ids'].to(config.device)
            padding_mask = batch['attention_mask'].to(config.device)
            targets = input_ids[:, 1:].contiguous()  # Shift targets by 1
            
            #Need to remove the last token from inputs and padding_mask
            inputs = input_ids[:, :-1].contiguous()  
            padding_mask = padding_mask[:, :-1].contiguous() 
            
            optimizer.zero_grad()
            outputs = gpt(inputs, padding_mask)
        
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            epoch_steps += 1
            global_step += 1
            
            if batch_idx % tc.log_interval == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                
                # Log to wandb
                wandb.log({
                    "batch_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch + 1,
                    "batch": batch_idx,
                    "global_step": global_step
                })
                
            del input_ids
            del padding_mask
        
        avg_loss = total_loss / len(trainloader)
        print(f'Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}')

        test_loss = validate_model(gpt, testloader, criterion)
        wandb.log({
            "epoch_train_loss": avg_loss,
            "epoch_test_loss": test_loss,
            "epoch": epoch + 1,
            "epoch_steps": epoch_steps
        })
        
        if (epoch + 1) % tc.save_interval == 0:
            checkpoint_path = f'gpt_checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': gpt.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config,
                'train_config': tc
            }, checkpoint_path)
            print(f'Checkpoint saved for epoch {epoch+1}')
            
            wandb.save(checkpoint_path)
            
        

    final_model_path = 'gpt_final_weights.pth'
    torch.save(gpt.state_dict(), final_model_path)
    print('Final model weights saved as gpt_final_weights.pth')
    
    wandb.save(final_model_path)
    
    wandb.log({
        "final_loss": avg_loss,
        "total_epochs": tc.epochs,
        "total_steps": global_step
    })

if __name__ == "__main__":
    wandb.login()
    wandb.init(
        project="gpt2",
        name="gpt2-tinystories-training",
    )
    
    c = config()
    tc = trainConfig()
    gpt = GPT(config=c)
    
    tc.epochs = 2
    
    trainloader, testloader = create_trainloader(tc)
    train_model(gpt, trainloader, testloader, tc)
    
    wandb.finish()
    
