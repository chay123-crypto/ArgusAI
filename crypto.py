from cryptography.fernet import Fernet
from helper import retry
from langsmith import traceable
from tools import critical_info

@traceable(name='anonymiser')
@retry(max_attempts=3,delay=3,backoff=3,exceptions=(Exception,))
def anonymiser(llm,profiler,df):
    key=Fernet.generate_key()
    f=Fernet(key)
    @retry(max_attempts=3,delay=3,backoff=3,exceptions=(Exception,))
    def get_critical():
        critical=critical_info(llm,profiler)
        return critical
    critical=get_critical()
    if not critical:
        raise ValueError("PII detection returned empty")
    anonymized_df=df.copy()
    mapping_log={}

    for col in critical:
        if col in df.columns.to_list():
            unique_vals=df[col].unique()
            mapping={}

            for val in unique_vals:
                encoded=f.encrypt(str(val).encode())
                display_id=f'ID-{encoded.decode()[9:15]}'
                dmap={'full':encoded.decode(),'display':display_id}  
                mapping[str(val)]=dmap 
            display_map={v: mapping[str(v)]['display'] for v in unique_vals}  
            anonymized_df[col]=anonymized_df[col].map(display_map)           
            mapping_log[col] = mapping
    return mapping_log,anonymized_df,key.decode()             

def decryption(key,mapping_log):
    reverse_map={}
    f=Fernet(key)
    for col,mapping in mapping_log.items():
        for encoded,displayed in mapping.items():
            try:
              decrypted=f.decrypt(displayed['full'].encode()).decode()
              reverse_map[displayed['display']]=decrypted
            except:
              reverse_map[displayed['display']]=str(encoded)
              
    return reverse_map  