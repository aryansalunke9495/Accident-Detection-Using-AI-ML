from tkinter import *
import os
from PIL import ImageTk, Image
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import cv2
import datetime
from tensorflow.keras.models import load_model
from twilio.rest import Client

SID = 'AC89330e3dcbfe37a432e15c111827bdf7'
AUTH_TOKEN = '7798334ceecae9e254dee109e543e383'

cl = Client(SID, AUTH_TOKEN)

# Function to play alert sound on Linux
def play_alert_sound():
    os.system('play -nq -t alsa synth 1 sine 2500 || aplay -q /dev/null 2>/dev/null')

home = Tk()
home.title("Accident Detection System")

img_path = "images/home.jpg"
img = Image.open(img_path)
img = ImageTk.PhotoImage(img)
panel = Label(home, image=img)
panel.pack(side="top", fill="both", expand="yes")

screen_width = home.winfo_screenwidth()
screen_height = home.winfo_screenheight()
lt = [screen_width, screen_height]

a = str(lt[0] // 2 - 450)
b = str(lt[1] // 2 - 320)
home.geometry("900x653+" + a + "+" + b)
home.resizable(0, 0)

file = ''

# Updated model path to use relative path
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'model.h5')
model = load_model(model_path)

def Exit():
    global home
    result = messagebox.askquestion("Accident Detection System", 'Are you sure you want to exit?', icon="warning")
    if result == 'yes':
        home.destroy()
        exit()
    else:
        messagebox.showinfo('Return', 'You will now return to the main screen')


def realtime():
    count = 0
    cap = cv2.VideoCapture(0)
    
    classes = ['No Accident', 'Accident']
    while 1:
            ret, img = cap.read()
            if not ret:
                messagebox.showerror("Error", "Cannot access camera. Please check camera connection.")
                break
                
            imgc = img.copy()
            img = cv2.resize(img,(224,224))
            img = img.reshape(1,224,224,3)/255.0  # Fixed reshape dimensions
            pred = model.predict(img)
            txt = classes[pred.argmax()]  # Fixed prediction indexing
            if 'No' not in txt:
                    if count>=50:
                            print("Accident Detected")
                            txt = "Accident Detected at "+str(datetime.datetime.now())
                            message = """\
                            Subject: Alert !!! Accident
                            """+txt
                            cl.messages.create(body=message, from_='+18317062001', to='+919460060699')
                            play_alert_sound()  # Using Linux compatible sound function
                            count=0
                    cv2.putText(imgc,txt, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,100,255),2)
                    count+=1
            else:
                    cv2.putText(imgc,txt, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,255,100),2)
                    
            cv2.imshow('Car Accident Detection',imgc)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                    break
            
    cap.release()
    cv2.destroyAllWindows()

def browse():
    global file, l1
    try:
        l1.destroy()
    except:
        pass
    file = askopenfilename(initialdir=os.getcwd(), title="Select video",
                           filetypes=(("images", ".mp4"), ("images", ".avi"), ("images", ".mkv")))
    if not file:
        return
        
    count = 0
    cap = cv2.VideoCapture(file)
    
    classes = ['No Accident', 'Accident']
    while 1:
            ret, img = cap.read()
            if not ret:
                break
                
            imgc = img.copy()
            img = cv2.resize(img,(224,224))
            img = img.reshape(1,224,224,3)/255.0  # Fixed reshape dimensions
            pred = model.predict(img)
            txt = classes[pred.argmax()]  # Fixed prediction indexing
            if 'No' not in txt:
                    if count>=50:
                            print("Accident Detected")
                            txt = "Accident Detected at "+str(datetime.datetime.now())
                            message = """\
                            Subject: Alert !!! Accident
                            """+txt
                            cl.messages.create(body=message, from_='+18317062001', to='+919460060699')
                            play_alert_sound()  # Using Linux compatible sound function
                            count=0
                    cv2.putText(imgc,txt, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,100,255),2)
                    count+=1
            else:
                    cv2.putText(imgc,txt, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,255,100),2)
                    
            cv2.imshow('Car Accident Detection',imgc)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                    break
            
    cap.release()
    cv2.destroyAllWindows()


def about():
    about = Toplevel()
    about.title("Accident Detection System")
    img_path_about = "images/about.jpg"
    img_about = Image.open(img_path_about)
    img_about = ImageTk.PhotoImage(img_about)
    panel_about = Label(about, image=img_about)
    panel_about.pack(side="top", fill="both", expand="yes")

    a_about = str(lt[0] // 2 - 450)
    b_about = str(lt[1] // 2 - 320)
    about.geometry("900x653+" + a_about + "+" + b_about)
    about.resizable(0, 0)
    about.mainloop()


photo = Image.open("images/1.png")
img2 = ImageTk.PhotoImage(photo)
b1 = Button(home, highlightthickness=0, bd=0, activebackground="#2b4b47", image=img2, command=browse)
b1.place(x=0, y=209)

photo = Image.open("images/2.png")
img3 = ImageTk.PhotoImage(photo)
b2 = Button(home, highlightthickness=0, bd=0, activebackground="#2b4b47", image=img3, command=realtime)
b2.place(x=0, y=282)

photo = Image.open("images/3.png")
img4 = ImageTk.PhotoImage(photo)
b3 = Button(home, highlightthickness=0, bd=0, activebackground="#2b4b47", image=img4, command=about)
b3.place(x=0, y=354)

photo = Image.open("images/4.png")
img5 = ImageTk.PhotoImage(photo)
b4 = Button(home, highlightthickness=0, bd=0, activebackground="#2b4b47", image=img5, command=Exit)
b4.place(x=0, y=426)

home.mainloop()

