import cv2
import numpy as np
import time
import os
import socket
import threading
import speech_recognition as sr

# --- Network Setup for Messaging ---
PORT = 6000
MODE = input("Enter 's' to host or 'c' to connect for messenger: ")
if MODE.lower() == 's':
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("", PORT))
    server.listen(1)
    print(f"Hosting messenger on port {PORT}, waiting for connection...")
    conn, addr = server.accept()
    print(f"Connected by {addr}")
elif MODE.lower() == 'c':
    peer_ip = input("Enter host IP address: ")
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.connect((peer_ip, PORT))
    print(f"Connected to {peer_ip}:{PORT}")
else:
    print("Invalid mode, exiting.")
    exit(1)

# Speech recognizer for voice transcription
def init_recognizer():
    r = sr.Recognizer()
    with sr.Microphone() as mic:
        r.adjust_for_ambient_noise(mic)
    return r
recognizer = init_recognizer()

# Chat history
chat_history = []
input_buffer = ''
new_msg = False

# Thread to receive messages
def listen_thread():
    global chat_history, new_msg
    while True:
        try:
            data = conn.recv(1024)
            if not data:
                break
            chat_history.append(f"Peer: {data.decode()}")
            new_msg = True
        except:
            break
threading.Thread(target=listen_thread, daemon=True).start()

# UI App States
STATE_MENU, STATE_CALC, STATE_CAMERA, STATE_GALLERY, STATE_MSG = range(5)
state = STATE_MENU

# Hover config
hovered = None
hover_start = 0
HOVER_TIME = 1.5

# Calculator state
calc_labels = ['7','8','9','/','4','5','6','*','1','2','3','-','C','0','=','+']
current_expr = ''

# Capture storage
dir_path = 'captures'
if not os.path.exists(dir_path): os.makedirs(dir_path)
captures = []

# Pointer detection HSV
detect_lower = np.array([20,100,100], dtype=np.uint8)
detect_upper = np.array([30,255,255], dtype=np.uint8)

# Draw button helper
def draw_button(img, rect, label, active=False):
    x1,y1,x2,y2 = rect
    bg = (50,50,50) if not active else (80,80,80)
    cv2.rectangle(img,(x1,y1),(x2,y2),bg,-1)
    cv2.rectangle(img,(x1,y1),(x2,y2),(180,180,250),2)
    cv2.putText(img,label,(x1+10,y1+(y2-y1)//2+5), cv2.FONT_HERSHEY_SIMPLEX,0.7,(240,240,240),2)

# Video capture for UI
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        frame = np.zeros((480,640,3), dtype=np.uint8)
    else:
        frame = cv2.flip(frame,1)
    h,w = frame.shape[:2]
    ui = frame.copy()
    t = time.time()

    # Title bar + Back
    bar_h = int(h*0.1)
    cv2.rectangle(ui,(0,0),(w,bar_h),(30,30,30),-1)
    titles = ['Main Menu','Calculator','Camera','Gallery','Messenger']
    cv2.putText(ui,titles[state],(20,bar_h-20),cv2.FONT_HERSHEY_SIMPLEX,1,(200,200,200),2)
    back_rect = (10, 10, int(w*0.3), bar_h-10)
    if state != STATE_MENU:
        draw_button(ui, back_rect, 'Back', hovered=='b')

    # Pointer detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, detect_lower, detect_upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations=2)
    mask = cv2.GaussianBlur(mask,(7,7),0)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pointer = None
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c)>1000:
            M = cv2.moments(c)
            if M['m00']:
                pointer = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
                cv2.circle(ui,pointer,10,(0,255,255),-1)
    if state == STATE_MENU:
        apps = [('Calc',STATE_CALC),('Cam',STATE_CAMERA),('Gal',STATE_GALLERY),('Msg',STATE_MSG)]
        bw = int(w*0.2); bh = int(h*0.15)
        gap = (w - 4*bw)//5; y0 = bar_h + 20
        sel = None
        for i,(lbl,st) in enumerate(apps):
            x = gap + i*(bw+gap)
            rect = (x,y0,x+bw,y0+bh)
            draw_button(ui, rect, lbl, hovered==i)
            if pointer and x<pointer[0]<x+bw and y0<pointer[1]<y0+bh:
                sel = i
        if sel is not None:
            if hovered==sel and t-hover_start>HOVER_TIME:
                state = apps[sel][1]; hovered=None
            elif hovered!=sel:
                hovered=sel; hover_start=t
        else:
            hovered=None

    elif state == STATE_CALC:
        # Expression display
        expr_h = int(h*0.2)
        cv2.rectangle(ui, (10, bar_h+10), (w-10, bar_h+expr_h), (30,30,30), -1)
        cv2.putText(ui, current_expr, (20, bar_h+expr_h-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (240,240,240),2)
        # Calculator grid
        grid_h = h-bar_h-expr_h-30
        bw = (w-20)//4; bh = grid_h//4
        sel = None
        for idx,lab in enumerate(calc_labels):
            r,c = divmod(idx, 4)
            rect = (10+c*bw, bar_h+expr_h+20+r*bh, 10+c*bw+bw-5, bar_h+expr_h+20+r*bh+bh-5)
            draw_button(ui, rect, lab, hovered==('c',idx))
            if pointer and rect[0]<pointer[0]<rect[2] and rect[1]<pointer[1]<rect[3]:
                sel = ('c', idx)
        # Back
        if pointer and sel is None and back_rect[0]<pointer[0]<back_rect[2] and back_rect[1]<pointer[1]<back_rect[3]:
            sel = 'b'
        if sel:
            if hovered==sel and t-hover_start>HOVER_TIME:
                if sel=='b': state=STATE_MENU
                else:
                    L = calc_labels[sel[1]]
                    if L=='C': current_expr=''
                    elif L=='=':
                        try: current_expr=str(eval(current_expr))
                        except: current_expr='Error'
                    else: current_expr+=L
                hovered=None
            elif hovered!=sel:
                hovered=sel; hover_start=t
        else:
            hovered=None

    elif state == STATE_CAMERA:
        ph = int(h*0.6); pw = int(w*0.6)
        win = cv2.resize(frame, (pw,ph))
        ui[bar_h+20:bar_h+20+ph, (w-pw)//2:(w+pw)//2] = win
        snap = ((w//2)-100, h-80, (w//2)+100, h-20)
        sel = None
        if pointer and snap[0]<pointer[0]<snap[2] and snap[1]<pointer[1]<snap[3]: sel='snap'
        draw_button(ui, snap, 'Snap', hovered=='snap')
        if pointer and sel is None and back_rect[0]<pointer[0]<back_rect[2] and back_rect[1]<pointer[1]<back_rect[3]: sel='b'
        if sel:
            if hovered==sel and t-hover_start>HOVER_TIME:
                if sel=='snap':
                    fname = f"captures/{int(t)}.png"
                    cv2.imwrite(fname, frame)
                    captures.append(fname)
                else:
                    state=STATE_MENU
                hovered=None
            elif hovered!=sel:
                hovered=sel; hover_start=t
        else:
            hovered=None

    elif state == STATE_GALLERY:
        # Gallery view
        gap = 10
        items = captures[-8:]
        cols = min(len(items), 4)
        # Thumbnail selection
        sel = None
        if cols:
            tw = (w - (cols + 1) * gap) // cols
            th = int(tw * 0.75)
            for i, f in enumerate(items):
                r, c = divmod(i, cols)
                x = gap + c * (tw + gap)
                y = bar_h + 20 + r * (th + gap)
                thumb = cv2.resize(cv2.imread(f), (tw, th))
                h_img, w_img = thumb.shape[:2]
                # Crop to UI bounds
                y_end = min(y + h_img, h)
                x_end = min(x + w_img, w)
                ui[y:y_end, x:x_end] = thumb[0:(y_end - y), 0:(x_end - x)]
                # Thumbnail hover/dwell
                if pointer and x < pointer[0] < x + w_img and y < pointer[1] < y + h_img:
                    if hovered == ('g', i) and (t - hover_start) > HOVER_TIME:
                        cv2.imshow('View', cv2.imread(f))
                        hovered = None
                    elif hovered != ('g', i):
                        hovered = ('g', i)
                        hover_start = t
        # Back button logic
        sel = None
        if pointer and back_rect[0] < pointer[0] < back_rect[2] and back_rect[1] < pointer[1] < back_rect[3]:
            sel = 'b'
        draw_button(ui, back_rect, 'Back', hovered == 'b')
        if sel == 'b':
            if hovered == 'b' and (t - hover_start) > HOVER_TIME:
                state = STATE_MENU
                hovered = None
            elif hovered != 'b':
                hovered = 'b'
                hover_start = t
        elif sel is None:
            hovered = None

    elif state == STATE_MSG:
        # Chat area
        chat_h = int(h*0.5)
        cv2.rectangle(ui,(10,bar_h+20),(w-10,bar_h+20+chat_h),(30,30,30),-1)
        y0,dy = bar_h+50,25
        for i,msg in enumerate(chat_history[-int(chat_h/dy):]):
            cv2.putText(ui,msg,(20,y0+i*dy),cv2.FONT_HERSHEY_SIMPLEX,0.6,(240,240,240),1)
        # Input box
        ib_y,ib_h = bar_h+30+chat_h, int(h*0.15)
        cv2.rectangle(ui,(10,ib_y),(w-10,ib_y+ib_h),(50,50,50),-1)
        cv2.putText(ui,input_buffer,(20,ib_y+30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(240,240,240),2)
        # On-screen keyboard
        keys = [list("QWERTYUIOP"), list("ASDFGHJKL"), list("ZXCVBNM")]
        space_key=' '; back_key='<-'
        k_rows = len(keys); key_h = int(ib_h/(k_rows+1))
        for r,row in enumerate(keys):
            n=len(row); key_w=int((w-20)/n)
            y = ib_y+ib_h + r*key_h + 10
            for c,key in enumerate(row):
                x=10+c*key_w; rect=(x,y,x+key_w-5,y+key_h-5)
                draw_button(ui,rect,key,hovered==('k',r,c))
                if pointer and x<pointer[0]<x+key_w and y<pointer[1]<y+key_h:
                    if hovered==('k',r,c) and t-hover_start>HOVER_TIME:
                        input_buffer+=key; hovered=None
                    elif hovered!=('k',r,c): hovered=('k',r,c); hover_start=t
        # Space & backspace
        y_sp=ib_y+ib_h + k_rows*key_h + 10; sp_w=int((w-40)/3)
        sp_rect=(10,y_sp,10+sp_w,y_sp+key_h)
        bk_rect=(20+sp_w,y_sp,20+2*sp_w,y_sp+key_h)
        draw_button(ui,sp_rect,'Space',hovered=='sp')
        draw_button(ui,bk_rect,back_key,hovered=='bk')
        sel=None
        if pointer:
            if sp_rect[0]<pointer[0]<sp_rect[2] and sp_rect[1]<pointer[1]<sp_rect[3]: sel='sp'
            elif bk_rect[0]<pointer[0]<bk_rect[2] and bk_rect[1]<pointer[1]<bk_rect[3]: sel='bk'
        if sel=='sp' and hovered=='sp' and t-hover_start>HOVER_TIME:
            input_buffer+=space_key; hovered=None
        elif sel=='bk' and hovered=='bk' and t-hover_start>HOVER_TIME:
            input_buffer=input_buffer[:-1]; hovered=None
        elif sel and hovered!=sel:
            hovered=sel; hover_start=t
        # Voice transcription button
        vt_w = int(w*0.15); vt_h=key_h
        vt_rect = (w-190-vt_w, ib_y+10, w-190, ib_y+10+vt_h)
        draw_button(ui, vt_rect, 'Voice', hovered=='vt')
        if pointer and vt_rect[0]<pointer[0]<vt_rect[2] and vt_rect[1]<pointer[1]<vt_rect[3]: sel='vt'
        if sel=='vt' and hovered=='vt' and t-hover_start>HOVER_TIME:
            try:
                with sr.Microphone() as mic:
                    audio = recognizer.listen(mic, timeout=5)
                    text = recognizer.recognize_google(audio)
                    input_buffer += ' ' + text
            except:
                pass
            hovered=None
        elif sel=='vt' and hovered!='vt': hovered='vt'; hover_start=t
        # Send button (enlarged)
        send_w = int(w*0.3)
        send_rect = (w-send_w-20, ib_y+10, w-20, ib_y+10+key_h)
        draw_button(ui,send_rect,'Send',hovered=='send')
        if pointer and send_rect[0]<pointer[0]<send_rect[2] and send_rect[1]<pointer[1]<send_rect[3]: sel='send'
        if sel=='send' and hovered=='send' and t-hover_start>HOVER_TIME:
            if input_buffer:
                conn.sendall(input_buffer.encode()); chat_history.append(f"Me: {input_buffer}"); input_buffer=''
            hovered=None
        elif sel=='send' and hovered!='send': hovered='send'; hover_start=t
        # Back
        if pointer and back_rect[0]<pointer[0]<back_rect[2] and back_rect[1]<pointer[1]<back_rect[3]: sel='b'
        draw_button(ui,back_rect,'Back',hovered=='b')
        if sel=='b' and hovered=='b' and t-hover_start>HOVER_TIME:
            state=STATE_MENU; hovered=None
        elif sel=='b' and hovered!='b': hovered='b'; hover_start=t


    cv2.imshow('SleekOS',ui)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
