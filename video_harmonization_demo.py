import torch
from models import S2CRNet,CFP
from torchvision.models import squeezenet1_1
import torch.nn.functional as F
import cv2

#########[settings]############## 
torch.set_default_tensor_type(torch.FloatTensor)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#############[create and load pretrained model]#############################
s2crnet = S2CRNet(squeezenet1_1(pretrained=False), stack=True).to(device)
s2crnet.load_state_dict(torch.load('S2CRNet_pretrained.pth',map_location=device)['state_dict'])

# VIDEO MATTING NETWORK 
from RobustVideoMatting.model import MattingNetwork
model = MattingNetwork('mobilenetv3').eval()  # or "resnet50"
model.load_state_dict(torch.load('rvm_mobilenetv3.pth',map_location=device))
model.to(device=device)

#### background image replacement
bac = torch.from_numpy(cv2.imread('background.jpg')[:,:,::-1]/255.).permute(2,0,1).unsqueeze(0).float()
bac = F.interpolate(bac,size=(480,640))

#### build label for s2crnet
label = torch.tensor([[0]])
label_oh = torch.zeros(1, 5).scatter_(1, label.view(label.shape[0], 1), 1).to(torch.float32).to(device)

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  
rec = [None] * 4

with torch.no_grad():
    while(True):

        # Capture the video frame
        ret, frame = vid.read()
        
        img = torch.from_numpy(frame[:,:,::-1]/255.).permute(2,0,1).unsqueeze(0).float().to(device) # foreground image

        #### segementation
        result, pha, *rec = model(img, *rec, 0.375)

        feeded = torch.cat([bac, 1-pha], dim=1).to(device)
        fore = torch.cat([img, pha], dim=1).to(device)

        outputs, param1, param2 = s2crnet(img, feeded, fore, label_oh, True)
        print('recalculate harmonization param')

        ### select the point in the foreground region
        mask = torch.where(pha > 0.5, torch.ones_like(pha), torch.zeros_like(pha)).bool().cpu()
        points = img.masked_select(mask).view(1,3,-1) # bs x c x point
        outputs = CFP(CFP(points,param1.view(1,3,1,64),64),param2.view(1,3,1,64),64).cpu()

        bacx = bac.clone().cpu()
        bacx[:,:,mask.squeeze()] = outputs
        bacx = bacx * pha + bac * (1-pha)
        img2 = img * pha + bac * (1-pha)

        output = torch.cat([img, img2, bacx],dim=3)

        cv2.imshow('frame', cv2.cvtColor(output[0].permute(1,2,0).numpy(),cv2.COLOR_RGB2BGR))
      
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
