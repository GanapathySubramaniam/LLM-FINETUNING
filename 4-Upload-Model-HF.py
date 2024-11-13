from huggingface_hub import login,create_repo,HfApi
import torch 

def upload_model_to_hf(model_output_dir,username,new_model_name):
  torch.cuda.empty_cache()
  login(token='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')
  create_repo(new_model_name)
  api = HfApi()
  api.upload_folder(
      folder_path=model_output_dir,
      repo_id=f"{username}/{new_model_name}",
      repo_type="model"
  )


# if __name__=="__main__":
#   upload_config={
#       'username':'Ganapathy0112357',
#       'new_model_name':'sample_model_123456',
#       'model_output_dir':output_path}
#   upload_model_to_hf(**upload_config)
