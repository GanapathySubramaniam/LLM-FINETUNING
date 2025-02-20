from huggingface_hub import login,create_repo,HfApi
import torch 

def upload_model_to_hf(model_output_dir,username,new_model_name):
  torch.cuda.empty_cache()
  login(token='hfXXXX')
  create_repo(new_model_name)
  api = HfApi()
  api.upload_folder(
      folder_path=model_output_dir,
      repo_id=f"{username}/{new_model_name}",
      repo_type="model"
  )

