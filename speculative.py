import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import yaml



# reading config files
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)



# setting up models and tokenizer
draft_model = AutoModelForCausalLM.from_pretrained(config['models']['draft_model'],torch_dtype=torch.float16).to("cuda").eval()
target_model = AutoModelForCausalLM.from_pretrained(config['models']['target_model'],torch_dtype=torch.float16).to("cuda").eval()
tokenizer = AutoTokenizer.from_pretrained(config['models']['tokenizer'])



# setting up hyper parameters
max_len = config['generation']['max_length']
gamma = config['generation']['gamma']



# noting <EOS> token id to end inference when found
eos_id=tokenizer.eos_token_id



# SPECULATIVE SAMPLING FUNCTION (can only used for batch_size==1)
@torch.no_grad()
def speculative_sampling(prompt):
    '''This function takes the raw prompt and then generates the output through speculative sampling along with the acceptance rate and inference speed'''
    
    
    # tokenizing the input
    prefix=tokenizer(prompt,return_tensors='pt').to('cuda').input_ids


    # noting down some parameters
    seq_len=prefix.shape[1]
    T=seq_len+max_len
    drafts_proposed=0
    drafts_accepted=0
    

    # Starting time
    torch.cuda.synchronize() if device=="cuda" else None
    t1=time.time()



    while(prefix.shape[1]<T):

        x=prefix
        n=prefix.shape[1]

        # generating /gamma tokens from draft model
        for _ in range(gamma):
            q=F.softmax(draft_model(x).logits,dim=2)
            x=torch.concat((x,torch.argmax(q[:,-1,:],dim=1).reshape(1,1)),dim=1)
            drafts_proposed+=1


        # generating logits from target model in one pass
        p=F.softmax(target_model(x).logits,dim=2)


        # iterating on tokens to reject or accept them
        all_accept=True
        for i in range(gamma):
            prob=torch.rand(1).to('cuda')

            # accepting the draft
            if(prob<=min(1,p[:,n-1+i,:][0,n+i]/q[:,n-1+i,:][0,n+i])):
                prefix=torch.concat((prefix,x[0,n+i].reshape(1,1)),dim=1)
                drafts_accepted+=1
                

                # emergency landing if EOS token is found
                if(x[0,n+i]==eos_id):
                    torch.cuda.synchronize() if device=="cuda" else None
                    t2=time.time()
                    acceptance_rate=drafts_accepted*100/drafts_proposed
                    inference_speed=(prefix.shape[1]-seq_len)/(t2-t1)
                    return prefix,acceptance_rate,inference_speed

            
            # rejecting the draft and sampling from the target distribution
            else:

                all_accept=False


                # sampling from (p-q) distriubtion
                d=p[:,n-1+i,:]-q[:,n-1+i,:]


                # scaling and taking norm
                d=torch.where(d>0,d,0)
                d_sum=torch.sum(d,dim=1)
                d=d/(d_sum+1e-10)


                prefix=torch.concat((prefix,torch.argmax(d).reshape(1,1)),dim=1)

                break



        # if all drafts are accepted, sampling a token from the target distribution
        if(all_accept==True):
            prefix=torch.concat((prefix,torch.argmax(p[:,-1,:]).reshape(1,1)),dim=1)


    # this is assuming no EOS token falls in the generation
    torch.cuda.synchronize() if device=="cuda" else None
    t2=time.time()
    acceptance_rate=drafts_accepted*100/drafts_proposed
    inference_speed=(prefix.shape[1]-seq_len)/(t2-t1)


    # returning the outputs
    return prefix,acceptance_rate,inference_speed


    

