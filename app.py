import streamlit as st
import os
from PIL import Image
import cv2
import datetime
import numpy as np
import tempfile
from tensorflow.keras.models import load_model
from twilio.rest import Client

# Set page configuration
st.set_page_config(
    page_title="Accident Detection System",
    page_icon="🚨",
    layout="wide",
)

# Twilio credentials
SID = 'AC89330e3dcbfe37a432e15c111827bdf7'
AUTH_TOKEN = '7798334ceecae9e254dee109e543e383'
cl = Client(SID, AUTH_TOKEN)

# Function to play alert sound (not used in Streamlit version)
def play_alert_sound():
    os.system('play -nq -t alsa synth 1 sine 2500 || aplay -q /dev/null 2>/dev/null')

# Load model
@st.cache_resource
def load_accident_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'model.h5')
    model = load_model(model_path)
    
    # Print model info for debugging (console only, not displayed in UI)
    print("Model input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)
    
    return model

# Model inspection function
def inspect_model(model):
    """Display model architecture and input/output shapes"""
    st.subheader("Model Architecture")
    
    try:
        # Display input/output shapes
        st.write(f"**Input Shape:** {model.input_shape}")
        st.write(f"**Output Shape:** {model.output_shape}")
        
        # Display model summary
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        
        st.text("Model Summary:")
        for line in summary_lines:
            st.text(line)
            
    except Exception as e:
        st.error(f"Error inspecting model: {str(e)}")
        
    # Test input shapes
    st.subheader("Testing Different Input Shapes")
    
    test_shapes = [
        (1, 224),           # Flattened 224 features
        (1, 224, 224, 1),   # Grayscale image
        (1, 224, 224, 3),   # RGB image  
        (1, 15, 15),        # Small image that flattens to ~224
        (1, 16, 14),        # Image that flattens to exactly 224
    ]
    
    for shape in test_shapes:
        try:
            test_input = np.random.random(shape).astype(np.float32)
            output = model.predict(test_input, verbose=0)
            st.success(f"✓ Shape {shape} works! Output shape: {output.shape}")
            break  # Found working shape
        except Exception as e:
            st.warning(f"✗ Shape {shape} failed: {str(e)[:100]}...")
    
    return True

# Load images
@st.cache_data
def load_images():
    home_img = Image.open("images/home.jpg")
    about_img = Image.open("images/about.jpg")
    return home_img, about_img

# Enhanced prediction function with automatic input format detection
def predict_accident(model, frame):
    """
    Enhanced prediction function that handles different model input formats
    """
    # Preprocess the frame
    img = cv2.resize(frame, (224, 224))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    try:
        # Approach 1: Resize to get exactly 224 features (most likely for your model)
        img_small = cv2.resize(img_gray, (15, 15))
        img_flattened = img_small.flatten()[:224]  # Take first 224 elements
        img_input = img_flattened.reshape(1, 224) / 255.0
        
        # Try prediction
        pred = model.predict(img_input, verbose=0)
        
    except Exception:
        try:
            # Approach 2: Try different flattening - 16x14 = 224
            img_14x16 = cv2.resize(img_gray, (16, 14))
            img_input = img_14x16.flatten().reshape(1, 224) / 255.0
            pred = model.predict(img_input, verbose=0)
            
        except Exception:
            # Approach 3: Fallback to original format
            img_input = img_gray.reshape(1, 224, 224, 1) / 255.0
            pred = model.predict(img_input, verbose=0)
    
    # Get prediction
    pred = model.predict(img_input, verbose=0)
    
    # Handle different prediction formats
    if len(pred.shape) == 2 and pred.shape[1] == 2:
        # Binary classification with 2 outputs [no_accident_prob, accident_prob]
        accident_prob = pred[0][1]
        no_accident_prob = pred[0][0]
        
        # Use the class with higher probability
        if accident_prob > no_accident_prob:
            return 1, accident_prob  # Accident detected
        else:
            return 0, no_accident_prob  # No accident
    
    elif len(pred.shape) == 2 and pred.shape[1] == 1:
        # Single output (binary classification with sigmoid)
        prob = pred[0][0]
        
        # Threshold at 0.5 for binary classification
        if prob > 0.5:
            return 1, prob  # Accident detected
        else:
            return 0, 1 - prob  # No accident
    
    else:
        # Fallback: use argmax
        pred_index = np.argmax(pred)
        confidence = np.max(pred)
        return pred_index, confidence

# Main function
def main():
    model = load_accident_model()
    home_img, about_img = load_images()
    
    # Create sidebar with navigation options
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Accident Detection (Video)", "Accident Detection (Camera)", "About"])
    
    if page == "Home":
        display_home(home_img)
    elif page == "Accident Detection (Video)":
        accident_detection_video(model)
    elif page == "Accident Detection (Camera)":
        accident_detection_camera(model)
    elif page == "About":
        display_about(about_img)

# Debug mode to test predictions
def debug_mode(model):
    st.title("Debug Mode - Test Model Predictions")
    
    uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mkv"])
    
    if uploaded_file is not None:
        if uploaded_file.type.startswith('image'):
            # Image debugging
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Convert PIL to OpenCV format
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Get prediction with debugging
            pred_class, confidence = predict_accident(model, frame, debug=True)
            
            classes = ['No Accident', 'Accident']
            result = classes[pred_class]
            
            st.write(f"**Prediction: {result}**")
            st.write(f"**Confidence: {confidence:.4f}**")
            
        elif uploaded_file.type.startswith('video'):
            # Video debugging - analyze first few frames
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            frame_count = 0
            
            st.write("Analyzing first 5 frames:")
            
            while frame_count < 5 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                st.write(f"--- Frame {frame_count + 1} ---")
                
                # Display frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(rgb_frame, caption=f"Frame {frame_count + 1}", width=300)
                
                # Get prediction with debugging
                pred_class, confidence = predict_accident(model, frame, debug=True)
                
                classes = ['No Accident', 'Accident']
                result = classes[pred_class]
                
                st.write(f"**Prediction: {result}**")
                st.write(f"**Confidence: {confidence:.4f}**")
                st.write("---")
                
                frame_count += 1
            
            cap.release()
            os.unlink(tfile.name)

# Home page
def display_home(home_img):
    st.title("Accident Detection System")
    st.image(home_img, use_container_width=True)
    st.markdown("""
    ## Welcome to Accident Detection System
    
    This application uses machine learning to detect accidents in videos and from camera feeds.
    
    ### Features:
    - Detection of accidents in uploaded videos
    - Real-time accident detection using camera
    - SMS alerts when accidents are detected (using Twilio)
    - Emergency stop functionality during processing
    
    Use the sidebar to navigate through the application.
    """)

# About page
def display_about(about_img):
    st.title("About Accident Detection System")
    st.image(about_img, use_container_width=True)
    st.markdown("""
    ## About This Project
    
    This application uses a trained deep learning model to detect accidents in video footage.
    
    When an accident is detected, the system can send SMS alerts to emergency contacts.
    
    ### Technologies Used:
    - TensorFlow for machine learning
    - OpenCV for video processing
    - Streamlit for the user interface
    - Twilio for SMS alerts
    
    ### Model Information:
    The model expects input images of size 224x224 pixels in grayscale format.
    It outputs probabilities for accident detection.
    """)

# Video accident detection function
def accident_detection_video(model):
    st.title("Accident Detection from Video")
    
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mkv"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        # Video processing options
        st.subheader("Processing Options")
        phone_number = st.text_input("Alert Phone Number (with country code)", "+919460060699")
        threshold = st.slider("Consecutive Accident Frames Threshold", 1, 20, 5)
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
        
        # Create columns for control buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            start_button = st.button("Start Processing")
        
        with col2:
            continue_button = st.button("Continue")
            
        # State management for video processing
        if 'processing_paused' not in st.session_state:
            st.session_state.processing_paused = False
        
        if 'current_frame' not in st.session_state:
            st.session_state.current_frame = 0
            
        if 'accident_detected' not in st.session_state:
            st.session_state.accident_detected = False
            
        # Show accident status if detected
        if st.session_state.accident_detected:
            st.error("🚨 ACCIDENT DETECTED - Processing stopped for safety review")
            st.info("Use 'Continue' button to resume processing or 'Start Processing' to restart from beginning")
            
        if start_button:
            # Reset ALL state
            st.session_state.processing_paused = False
            st.session_state.current_frame = 0
            st.session_state.accident_detected = False
            process_video_with_pause(tfile.name, model, phone_number, threshold, confidence_threshold)
            
        if continue_button and st.session_state.processing_paused:
            st.session_state.processing_paused = False
            # Don't reset accident_detected - keep it for reference
            process_video_with_pause(tfile.name, model, phone_number, threshold, confidence_threshold, start_frame=st.session_state.current_frame)

# Function to process the video file with pause functionality
def process_video_with_pause(video_path, model, phone_number, threshold, confidence_threshold, start_frame=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file")
        return
    
    # Create emergency stop button
    emergency_stop = st.empty()
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Get video information for progress tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create a placeholder for video frames
    frame_placeholder = st.empty()
    
    # Detection variables
    classes = ['No Accident', 'Accident']
    accident_count = 0
    frame_count = start_frame
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    while cap.isOpened():
        # Check for emergency stop
        with emergency_stop:
            if st.button("STOP", key=f"emergency_{frame_count}", type="primary"):
                st.session_state.processing_paused = True
                st.session_state.current_frame = frame_count
                break
        
        if st.session_state.processing_paused:
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 3rd frame for speed
        if frame_count % 3 == 0:
            # Update progress
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames} | Accident count: {accident_count}/{threshold}")
            
            # Get prediction using enhanced function
            pred_class, confidence = predict_accident(model, frame)
            txt = classes[pred_class]
            
            # Draw on frame
            if pred_class == 1 and confidence >= confidence_threshold:  # Accident detected
                accident_count += 1
                cv2.putText(frame, f"{txt} ({confidence:.2f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"ALERT COUNT: {accident_count}/{threshold}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if accident_count >= threshold:
                    # Accident confirmed - STOP EVERYTHING
                    accident_time = datetime.datetime.now()
                    st.error(f"🚨 ACCIDENT DETECTED at {accident_time}")
                    
                    # Send SMS alert immediately
                    message = f"EMERGENCY ALERT: Accident detected at {accident_time.strftime('%Y-%m-%d %H:%M:%S')}. Location: Video Analysis System. Please respond immediately."
                    
                    try:
                        cl.messages.create(body=message, from_='+18317062001', to=phone_number)
                        st.success("✅ Emergency SMS sent successfully!")
                        
                        # Also try to send a follow-up SMS
                        follow_up = f"This is an automated accident detection alert. Frame: {frame_count}. Time: {accident_time.strftime('%H:%M:%S')}"
                        cl.messages.create(body=follow_up, from_='+18317062001', to=phone_number)
                        
                    except Exception as e:
                        st.error(f"❌ CRITICAL: Failed to send emergency SMS: {str(e)}")
                        # Visual emergency alert without balloons
                        st.warning("⚠️ SMS FAILED - Check phone number and network connection!")
                    
                    # FORCE STOP - Set multiple stop conditions
                    st.session_state.processing_paused = True
                    st.session_state.current_frame = frame_count
                    st.session_state.accident_detected = True
                    
                    # Display accident frame prominently
                    st.subheader("🚨 ACCIDENT DETECTED - PROCESSING STOPPED 🚨")
                    accident_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(accident_frame, caption=f"Accident detected at frame {frame_count}", use_container_width=True)
                    
                    # Clean up and exit
                    cap.release()
                    progress_bar.progress(1.0)
                    status_text.text(f"🚨 STOPPED: Accident detected at frame {frame_count}")
                    return  # EXIT FUNCTION COMPLETELY
            else:
                # No accident or low confidence
                accident_count = max(0, accident_count - 1)  # Decay counter
                cv2.putText(frame, f"{txt} ({confidence:.2f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if accident_count > 0:
                    cv2.putText(frame, f"Count: {accident_count}/{threshold}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
    
    # Clean up
    cap.release()
    progress_bar.progress(1.0)
    status_text.text("Video processing complete!")

# Camera accident detection function
def accident_detection_camera(model):
    st.title("Real-time Accident Detection (Camera)")
    
    # Phone number for alerts
    phone_number = st.text_input("Alert Phone Number (with country code)", "+919460060699")
    threshold = st.slider("Consecutive Accident Frames Threshold", 1, 20, 5)
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
    
    # Placeholder for the camera feed
    stframe = st.empty()
    
    # Add option to use test video instead of camera in environments with no camera access
    st.warning("Note: If you're running this in a cloud environment like GitHub Codespaces, camera access may not be available.")
    use_test_video = st.checkbox("Use test video instead of camera")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_button = st.button("Start Camera/Video")
        
    with col2:
        stop_button = st.button("Stop")
    
    # Create session state for camera processing
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
        
    if 'test_video_path' not in st.session_state:
        # Default to the included video if available, otherwise allow upload
        test_video_path = "/workspaces/hitesh/Accident Detection with UI/1.mp4"
        if os.path.exists(test_video_path):
            st.session_state.test_video_path = test_video_path
        else:
            st.session_state.test_video_path = None
    
    # Allow test video upload if default isn't available
    if use_test_video and not st.session_state.test_video_path:
        uploaded_file = st.file_uploader("Upload test video", type=["mp4", "avi", "mkv"])
        if uploaded_file:
            # Save the uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            st.session_state.test_video_path = tfile.name
    
    if start_button:
        st.session_state.camera_running = True
        
        if use_test_video and st.session_state.test_video_path:
            # Use test video instead of camera
            cap = cv2.VideoCapture(st.session_state.test_video_path)
            if not cap.isOpened():
                st.error(f"Error opening test video file: {st.session_state.test_video_path}")
            else:
                st.success("Using test video for detection")
                process_camera_feed(cap, model, stframe, phone_number, threshold, confidence_threshold)
        else:
            # Try to use camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot access camera. Please check camera connection or try using a test video instead.")
            else:
                process_camera_feed(cap, model, stframe, phone_number, threshold, confidence_threshold)
    
    if stop_button:
        st.session_state.camera_running = False
        st.info("Camera/Video stopped")

# Function to process camera feed
def process_camera_feed(cap, model, stframe, phone_number, threshold, confidence_threshold):
    classes = ['No Accident', 'Accident']
    accident_count = 0
    
    # Create a stop button container
    stop_container = st.empty()
    
    while st.session_state.camera_running:
        ret, frame = cap.read()
        if not ret:
            st.error("Error reading from camera/video")
            break
        
        # Get prediction using enhanced function
        pred_class, confidence = predict_accident(model, frame)
        txt = classes[pred_class]
        
        # Draw on frame
        if pred_class == 1 and confidence >= confidence_threshold:  # Accident detected
            accident_count += 1
            cv2.putText(frame, f"{txt} ({confidence:.2f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Count: {accident_count}/{threshold}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if accident_count >= threshold:
                # Accident confirmed - IMMEDIATE STOP AND ALERT
                accident_time = datetime.datetime.now()
                st.error(f"🚨 ACCIDENT DETECTED at {accident_time}")
                
                # Send emergency SMS immediately
                emergency_msg = f"🚨 EMERGENCY ALERT 🚨\nAccident detected at {accident_time.strftime('%Y-%m-%d %H:%M:%S')}\nLocation: Real-time Camera Feed\nIMMEDIATE RESPONSE REQUIRED"
                
                try:
                    # Send primary alert
                    cl.messages.create(body=emergency_msg, from_='+18317062001', to=phone_number)
                    st.success("✅ Emergency SMS sent!")
                    
                    # Send follow-up with timestamp
                    follow_up = f"Accident Alert #{datetime.datetime.now().strftime('%H%M%S')} - Camera feed stopped. Please check location immediately."
                    cl.messages.create(body=follow_up, from_='+18317062001', to=phone_number)
                    
                except Exception as e:
                    st.error(f"❌ CRITICAL: Failed to send emergency SMS: {str(e)}")
                    # Visual emergency alert without balloons
                    st.warning("⚠️ SMS FAILED - Check phone number and network connection!")
                
                # FORCE STOP camera feed
                st.session_state.camera_running = False
                
                # Display final accident frame
                st.subheader("🚨 ACCIDENT DETECTED - CAMERA STOPPED 🚨")
                rgb_frame_final = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(rgb_frame_final, caption=f"Accident detected at {accident_time}", use_container_width=True)
                
                # Clean up and exit
                cap.release()
                st.info("Camera feed stopped due to accident detection")
                return  # EXIT FUNCTION COMPLETELY
        else:
            # No accident or low confidence
            accident_count = max(0, accident_count - 1)  # Decay counter
            cv2.putText(frame, f"{txt} ({confidence:.2f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if accident_count > 0:
                cv2.putText(frame, f"Count: {accident_count}/{threshold}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb_frame, channels="RGB", use_container_width=True)
        
        # Check if stop button is pressed
        with stop_container:
            if st.button("Stop Camera", key=f"stop_{datetime.datetime.now().timestamp()}"):
                st.session_state.camera_running = False
                break
    
    # Clean up
    cap.release()
    st.info("Camera/Video stopped")

if __name__ == "__main__":
    main()
