# AI Usage

## Tool Used
Claude (Anthropic), used as a pair programmer for explaining the assignment,
writing and debugging code, and suggesting experiments.

## How AI Helped
1. **Understanding the task:** Explained the assignment requirements.
2. **Augmentation without data leakage:** Helped write augmentation code that loads the
   training folder twice, so validation images are never augmented. I verified this
   by checking that train/validation overlap was 0.
3. **Understanding several parts of the implementation:** Took help to write sections and 
helped me with the thorough understanding.

## A Questionable AI Suggestion
The AI's example Google Drive path did not match my real folder structure, which caused
a `FileNotFoundError`. I had to find and set the correct path myself.

## How I Verified Results
My first baseline run reached 97% validation accuracy after one epoch, which seemed too
high for a weak starter model. I questioned it, and the cause was a directory problem:
the dataset folders were at the wrong level, so the model was not seeing the 16 classes
correctly. After fixing the folders, I re-ran the baseline.
I also checked batch shapes, the train/validation overlap, and that the saved checkpoint
reproduced my test accuracy.

## A Decision I Made Myself
To begin with, the obvious choice was to start testing the epoch number and learning rate, then 
changed the greyscale with color and then data augmentation and finally pretrained imagenet model, as we heard a 
lot about imagenet and resnet and alexnet, it was a clear choice to try a pretrained model with the assignment.
