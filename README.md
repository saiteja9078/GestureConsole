# Gesture Console

Gesture Console is a macOS application that allows you to control games using physical gestures captured through your webcam. Using pose detection technology, the app transforms your body movements into keyboard inputs, creating an immersive gaming experience without specialized hardware.

## Features

- **Custom Gesture Training**: Train the app to recognize your specific movements
- **Multiple Game Profiles**: Create and save gesture profiles for different games
- **Key Press and Hold Support**: Configure gestures to either press or hold keyboard keys
- **Real-time Gesture Recognition**: Advanced pose detection with responsive feedback
- **Intelligent Motion Smoothing**: Prevents accidental inputs with temporal filtering
- **User-friendly Interface**: Simple two-tab design for training and playing

## Requirements

- macOS 11 (Big Sur) or later
- Python 3.7+ (if running from source)
- Webcam/camera access
- 4GB RAM recommended
- 2GB free disk space

## Installation

### Download the DMG for mac
1. Download the latest release from the [Releases page]([https://github.com/yourusername/gesture-console/releases](https://github.com/saiteja9078/GestureConsole/releases/tag/1.0.0))
2. Open the DMG file
3. Drag Gesture Console to your Applications folder
5. Run this command in your terminal (Replace /Applications/GestureConsole.app with your application path.)
```
xattr -d com.apple.quarantine /Applications/GestureConsole.app
 ```
6. Give permissions for accecebility and camera if not prompted when openeed.

### For windows versions click [here](https://github.com/vinith-369/GestureConsole/releases/tag/v1.0.1).

### Training Gestures
1. Select the "Train Gestures" tab
2. Enter a name for your game profile and click "Set Game"
3. Add gestures by selecting the action type (press or hold), entering the key, and clicking "Add Gesture"
4. Add an idle position by clicking "Add Idle Position"
5. Select the top 1st gesture from the list and click "Start Camera"
6. Click "Start Collection" to begin capturing poses for the selected gesture
7. Once all gestures are collected, click "Start Training" to create your model

### Playing Games
1. Select the "Play Game" tab
2. Choose your game profile from the dropdown
3. Click "Start Camera" to activate your webcam
4. Start your game and click "Start Playing" to begin gesture control
5. The app will display your current detected gesture and execute the corresponding key actions

## Development

Gesture Console is built using:
- Python 3.9.21
- PyQt5 for the user interface
- MediaPipe for pose detection
- scikit-learn for gesture classification
- OpenCV for camera handling and image processing

## Privacy

Gesture Console processes all video locally on your device. No data is sent to external servers or stored beyond what's needed for the gesture recognition model.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any issues or have questions, please open an issue on the GitHub repository.
