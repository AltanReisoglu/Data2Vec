from dataset import train_dataloaded,valid_dataloaded
from main_model import data2vec_base
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR,CosineAnnealingWarmRestarts
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.optim import Adam

model_teacher=data2vec_base(is_teacher=True,masking_ratio=0)
model_student=data2vec_base(is_teacher=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(model_teacher,model_student,train_dataloaded,valid_dataloaded):
    model_teacher=model_teacher.to("cuda")
    model_student=model_student.to("cuda")
    train_dataloaded=train_dataloaded
    valid_dataloaded=valid_dataloaded
    

    # Modelleri cihaza taşıyoruz
    num_epochs=40
    train_sets = len(train_dataloaded)
    warmup_epochs=5

    context_opt = torch.optim.AdamW(model_teacher.parameters(), lr=1e-4)
    predictor_opt = torch.optim.AdamW(model_student.parameters(), lr=1e-4)
    scheduler = get_cosine_schedule_with_warmup(predictor_opt,train_sets* warmup_epochs, train_sets * num_epochs)
    criterion = torch.nn.MSELoss()


    train_losses = []
    model_teacher.train()
    model_student.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_loss_val=0.0
        
        for idx, (x, _) in enumerate(tqdm(train_dataloaded, desc=f"Epoch {epoch+1}/{num_epochs}")):
            x = x.to(device)
            
            # context_model'den context, target ve target_indices elde ediliyor
            context= model_teacher(x)
            
            # target_model'den hedef temsiller (target representations) alınıyor
            context_student = model_student(x)

            loss = criterion(context, context_student)

            loss.backward()
            
            # Loss toplamına ekleme (skaler olarak)
            total_loss += loss.item()
            
            # Parametre güncellemesi
            context_opt.step()
            predictor_opt.step()
            
            context_opt.zero_grad()
            predictor_opt.zero_grad()
            if device == "cuda":
                torch.cuda.synchronize()
            
            # CUDA kullanıyorsak senkronizasyon
            
        for idx,(images,label) in enumerate(tqdm(valid_dataloaded, desc=f"Epoch {epoch+1}/{num_epochs}")):
            x = images.to(device)

            context = model_teacher(x)
            context_student = model_student(x)
            
            with torch.no_grad():
   
                loss_i = criterion(context, context_student)

                total_loss_val += loss_i.item()

                if device == "cuda":
                    torch.cuda.synchronize()
            # Öğrenme oranı güncellemesi
        scheduler.step()
        
        avg_loss = total_loss / len(train_dataloaded)
        avg_loss_v=total_loss_val/len(valid_dataloaded)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.6f} | Val Loss: {avg_loss_v:.6f}")
    
    print("Finished Training")
    return model_student,model_teacher

model_student,model_teacher = train_model(model_teacher,model_student,train_dataloaded,valid_dataloaded)