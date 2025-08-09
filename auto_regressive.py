import torch as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
import time



# reading config files
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)



# Loading model and tokenizer
target_model = AutoModelForCausalLM.from_pretrained(config['models']['target_model'],torch_dtype=nn.float16).to("cuda").eval()
tokenizer = AutoTokenizer.from_pretrained(config['models']['tokenizer'])



# setting up hyper parameters
max_len = config['generation']['max_length']
gamma = config['generation']['gamma']



# noting <EOS> token id to end inference when found
eos_id=tokenizer.eos_token_id


@torch.no_grad()
def auto_regressive(prompt):
    '''Generating response from target model'''


    # tokenizing the input
    prefix=tokenizer(prompt,return_tensors='pt').to('cuda').input_ids


    seq_len=prefix.shape[1]
    T=seq_len+max_len


    # Starting time
    torch.cuda.synchronize() if device=="cuda" else None
    t1=time.time()
    x=prefix


    while(x.shape[1]<T):
        
        probs=F.softmax(target_model(x).logits,dim=2)
        x=nn.concat((x,nn.argmax(probs[:,-1,:],dim=1).reshape(1,1)),dim=1)

        if(x[0,-1]==eos_id):
            torch.cuda.synchronize() if device=="cuda" else None
            t2=time.time()
            inference_speed=(x.shape[1]-seq_len)/(t2-t1)
            return x,inference_speed


    torch.cuda.synchronize() if device=="cuda" else None
    t2=time.time()
    inference_speed=(x.shape[1]-seq_len)/(t2-t1)
    return x,inference_speed

