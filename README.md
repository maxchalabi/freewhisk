# freewhisk
A global whisker movement tracking algorithm specialized for freely moving, untrimmed mice. 

There are multiple sophisticated algorithms out there for tracking the shape and displacement of whiskers in rodents, which they use to sense their tactile surroundings. These algorithms work best for head-fixed animals, where background noise and artefacts can be kep at a minimum. The mouse whisker system has indeed become a popular neuroscientific model to study tactile sensory processing and these whisker tracking algorithms are very helpful in linking whisker bending or whisker motor movement to neuronal signals.

Technological developments are starting to allow us to record neuronal activity in freely moving rodents, and with that we need ways to quantify whisking dynamics of these animals under conditions where 'conventional' whisker tracking systems can become unreliable. Unlike other whisker tracking algorithms which aim to identify the shapes and track the kinematics of individual whiskers, freewhisk provides a very basic approach to extract a global whisking envelope, without focusing on more precise measures like curvature.

![image](https://user-images.githubusercontent.com/47891330/194781755-fee6b3cf-6a7b-49b6-9da3-fd66385e7b67.png)

The method is particularly useful in conditions where we might be dealing with a lot of noise; uneven background illumination, paws moving in proxomity of the whiskers, uneven camera focus, different rows of whiskers moving in and out of focus as mice lift and rotate their head (whiskers also move in 3D space), whiskers interacting with an object, etc....

All we need for it to work is a video of a mouse taken from above, the tracked nose xy coordinates and head direction coordinates for the video.

