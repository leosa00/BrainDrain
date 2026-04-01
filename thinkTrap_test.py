# apg_test.py                                                                                                                                                                                                                                                                
import asyncio                                                                                                                                                                                                                                                                 import logging                                                                                                                                                                                                                                                               
from core.base_attack import APIFormat, TargetConfig                                                                                                                                                                                                                         
from attacks.think_trap import (                                                                                                                                                                                                                                           
    ThinkTrapConfig,                                                                                                                                                                                                                                                         
    ThinkTrapAPG,                                         
    load_surrogate_embeddings,                                                                                                                                                                                                                                               
    make_victim_fn,                                       
)
import secret
                                                                                                                                                                                                                                                                               
logging.basicConfig(level=logging.INFO)
                                                                                                                                                                                                                                                                               
target = TargetConfig(                                    
    base_url="https://api.deepseek.com",   # or whichever victim endpoint
    model="deepseek-reasoner",                                                                                                                                                                                                                                               
    api_format=APIFormat.OPENAI,
    api_key=secret.deepseek_api_key,                                                                                                                                                                                                                                         
    timeout=120.0,                                        
)                                                                                                                                                                                                                                                                            
                                                            
async def main():                                                                                                                                                                                                                                                            
    # Step 1 — load surrogate embeddings                  
    # Option A: from HuggingFace (slow, downloads weights)
    T, tok = load_surrogate_embeddings("meta-llama/Llama-2-7b-hf")                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                               
    # Option B: from a pre-saved .npy file (fast, after first run)                                                                                                                                                                                                           
    # np.save("llama2_embeddings.npy", T)   # save once                                                                                                                                                                                                                      
    # T, tok = load_surrogate_embeddings("", embeddings_path="llama2_embeddings.npy")                                                                                                                                                                                        
                                                                                                                                                                                                                                                                               
    cfg = ThinkTrapConfig(                                                                                                                                                                                                                                                   
        prompts_file="prompts/thinktrap_prompts.json",                                                                                                                                                                                                                       
        prompt_length=20,       # L                       
        latent_dim=20,          # m                                                                                                                                                                                                                                          
        cmaes_sigma=1.0,
        query_budget=200,       # number of victim API calls                                                                                                                                                                                                                 
        top_k_keep=10,                                                                                                                                                                                                                                                       
    )
                                                                                                                                                                                                                                                                               
    victim = make_victim_fn(target, max_tokens=4096, tokenizer=tok)                                                                                                                                                                                                          
   
    apg = ThinkTrapAPG(victim_fn=victim, T_surrogate=T, config=cfg, tokenizer=tok, seed=42)                                                                                                                                                                                  
    await apg.run()                                       
    apg.save("prompts/thinktrap_prompts.json")                                                                                                                                                                                                                               
    print(f"Saved {apg.cache if hasattr(apg, 'cache') else len(apg.best_prompts)} prompts")                                                                                                                                                                                  
                                                                                                                                                                                                                                                                               
asyncio.run(main())                                                                       