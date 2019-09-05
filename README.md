## Business Card Reader
Reads the given business card image using *image processing*. Extracted text then filtered out using *regex*:

 1. URL
 2. E-Mail
 3. Phone Number

*Text classifier* used to extract:

 4. Name
 5. Organization

The image is modified to get better results from the image processing. Few forms of noise reduction, binarization, and thresh is tested out in the driver.  You can use my business card image under *test_image/4.JPG* to test the driver.

**Libraries Needed**

 - cv2
 - numpy
 - re
 - spacy
 - displacy from spacy --> Optional
 - filter_re 

**How to Run?**

 - After installing the libraries using '*pip install libraryname*' (except filter_re), simply run the driver using '*python driver.py*'.
 - You can try different images and train your own *spacy* model for better accuracy.
 - Use *filter_re* for different *regex* filters. 
 - If you don't have GPU, you must comment out the line 110 so that it doesn't try to use your GPU.
 
**Future Work**
This is written to be part of a backend code for future projects. Current code is only good for testing. Improvements needed in the text classifier, and image processor. I plan to push the Flask API code which will use improved version of this code. 
 
