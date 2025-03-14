# 😎 Face Finder 🔍🤳

This is a cool computer program that can find faces in pictures and remember them! It's like teaching your computer to recognize your friends. ✨🧠💻

## 🌟 What This Program Can Do 🌟

- 👁️ Look at lots of pictures and find all the faces
- 🧠 Remember what each face looks like
- 🔄 Find pictures of the same person even if they look different (like smiling 😃 or frowning 😠)
- 🔎 Let you search for someone's face to see if they're in other pictures

## 🎮 How To Use It 🎮

You can talk to the program by typing special commands:

```
python main.py [options]
```

Here are the cool things you can tell it to do:

- `--save` 💾 : Tell the program to look at your pictures and remember the faces
- `--source FOLDER` 📁 : Tell it which folder has your pictures (if you don't say, it looks in a folder called "events")
- `--target PICTURE` 🎯 : Show it a picture of someone to find
- `--search` 🔍 : Ask it to look for the person in your picture
- `--show` 📺 : Ask it to show you the matches with pictures
- `--threads NUMBER` ⚡ : Make it work faster by using more brain power
- `--sequential` 🐢 : Make it work more carefully one picture at a time

### ✨ Examples ✨

1. To scan all your pictures and remember the faces:
   ```
   python main.py --save
   ```
   📸 → 💾

2. To scan pictures in your vacation folder:
   ```
   python main.py --save --source my_summer_vacation
   ```
   🏖️ → 💾

3. To find pictures of your friend from a photo:
   ```
   python main.py --target picture_of_my_friend.jpg --search
   ```
   👧 → 🔍

4. To find your friend AND see the matching pictures:
   ```
   python main.py --target picture_of_my_friend.jpg --search --show
   ```
   👧 → 🔍 → 📺

## 🧙‍♂️ How It Works 🪄

This program uses a special computer brain called "FaceNet" 🧠 that can look at a face and turn it into 128 numbers. These numbers describe the face - like how far apart the eyes are 👀 or how big the nose is 👃.

When the program remembers a face, it stores these numbers in a special box called a "database" 📦. Later, when you want to find someone, it compares their face numbers to all the saved face numbers and finds the closest matches! ✅

## 👁️ Face Detection Tools 🔍

The program can use different tools to find faces in pictures:

1. **RetinaFace** 🦅: This is super accurate but needs a powerful computer. 💪💻
2. **MTCNN** 🏆: This is pretty good and works well on most computers. 👍
3. **Dlib** 🚀: This is faster but might miss some faces. 💨
4. **OpenCV** ⚡: This is the fastest but not as good at finding all faces. 🏎️

## 🧠 Face Recognition Tools 🤔

After finding a face, the program uses these tools to remember it:

1. **FaceNet** 🌟: This is really good at remembering faces accurately. 💯
2. **ArcFace** 🏹: This is also very accurate. 🎯
3. **VGG-Face** 👤: This is not as good but works OK. 👌
4. **Facenet512** ⭐: This is a better version of FaceNet. 📈

## 🛠️ What You Need To Run This Program 🛠️

- A computer program called PostgreSQL 🐘 with something special added to it called pgvector 📊
- Some Python helper programs: 
  - 🐍 deepface
  - 🔌 psycopg2
  - 👁️ opencv-python
  - 📊 matplotlib
  - ⏱️ tqdm

Don't worry about all these fancy names - a grown-up can help you install them if you want to try the program! 👨‍👩‍👧‍👦 👩‍💻
