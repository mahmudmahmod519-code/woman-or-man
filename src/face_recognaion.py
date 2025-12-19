#init detect face
#get mask face
#fit model
#know model is man or woman

# one image -> have multable face

#know face
import cv2
from sklearn.tree import DecisionTreeClassifier
import skimage.feature as extract_feature_skimage
import skimage.filters as extract_filters
from ultralytics import YOLO
#man and woman


model = YOLO('../model/yolov9m-face-lindevs.pt')


mans_path='../data-set/mans'
womans_path='../data-set/womans'

model_ML=DecisionTreeClassifier()
FACE_SIZE=(64,64)

def prepare_data(main_path,type_y="man"):
  paths=os.listdir(main_path)
  # loop on image
  for path in paths:

    # read image
    image=cv2.imread(f'{main_path}/{path}')


    faces=know_face(image)

    # if masks = none this mean the image not have mask
    if len(faces[0].boxes.xyxy)==0:
      continue

    feataure_mask=[]
    output=[]

    # loop on face and prepare x and y for model
    for x1,y1,x2,y2 in faces[0].boxes.xyxy:
      mask=image[int(y1):int(y2),int(x1):int(x2)]
      mask=cv2.resize(mask, FACE_SIZE)
      feataure_mask.append(extract_feature(mask))#[[]]
      output.append(type_y)#['']

    yield feataure_mask,output#[[]],['']



#know face
def know_face(image):

  #prepare image for model YOLO
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  #add image for model
  faces = model(image_rgb)

  return faces


#extract feataure
def extract_feature(mask):
  #prepare mask gray to extract feataure
  gray_image=cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)

  filt_real,filt_image=extract_filters.gabor(gray_image,frequency=0.6)
  magnitude = np.sqrt(filt_real**2 + filt_image**2)

  get_texture=extract_feature_skimage.local_binary_pattern(gray_image,8,1)

  extract_histogram=extract_feature_skimage.hog(gray_image)

  features = np.concatenate([
    magnitude.flatten(),
    get_texture.flatten(),
    extract_histogram.flatten() if hasattr(extract_histogram, 'flatten') else np.array([extract_histogram])
  ])

  return features#[]


feataure_mask=[]#x
output=[]#y

iterat_man=prepare_data(mans_path,"man")

for x_man,array_of_man_name in iterat_man:
  feataure_mask.extend(x_man)
  output.extend(array_of_man_name)

iterat_woman=prepare_data(womans_path,"woman")

for x_woman,array_of_woman_name in iterat_woman:
  feataure_mask.extend(x_woman)
  output.extend(array_of_woman_name)


model_ML.fit(feataure_mask,output)



# output code
def predict_face(image):

  faces=know_face(image)

  feataure_mask=[]

  if(len(faces[0].boxes.xyxy)==0):
    return ["who i am ?"]

  # loop on face and prepare x and y for model
  for x1,y1,x2,y2 in faces[0].boxes.xyxy:
    mask=image[int(y1):int(y2),int(x1):int(x2)]
    mask=cv2.resize(mask, FACE_SIZE)
    feataure_mask.append(extract_feature(mask))

  return model_ML.predict(feataure_mask)



path=input("enter absolute path for video :")
if not path or path=='0':
  path=0
  
c=cv2.VideoCapture(path)
if not c.isOpened():
    exit()
        
while(True):
    
    c.grab()
    
    ret,image=c.retrieve()
    
    faces=know_face(image)

    text=predict_face(image)

    for index in range(len(faces[0].boxes.xyxy)):
      [x1,y1,x2,y2]=list(map(int,faces[0].boxes.xyxy[index]))
      # put predict here on rectangle
      cv2.rectangle(image,(x1,y1),(x2,y2),(255,255,255),2)
      cv2.putText(image,text[index],(x1-5,y1-5),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,255),2)


    
    cv2.imshow('camera1',image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

c.release()
cv2.destroyAllWindows()