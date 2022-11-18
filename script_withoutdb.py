import cv2
import argparse
import pandas as pd
import re
from PIL import Image
import os
from pyPdfToImg.pdf2image import convert_from_path
from pyImgToText.pytesseract import image_to_string, image_to_data
import math
import cv2
import numpy as np
from typing import Tuple, Union
from deskew import determine_skew
from xlwt import Workbook
import shutil
import streamlit as st

print_date=""
print_number=""
print_ammount=""
print_buyer=""
print_seller=""

#function to rotate and remove the skew from the image
def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + \
        abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + \
        abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


#function  to get the starting parameter and ending paramter for part 3 
def getstartend(total_amount, text):
    
    lines = text.split("\n")

    #remove the empty lines
    non_empty_lines = [line for line in lines if line.strip() != ""] 
    string_without_empty_lines = ""
    for line in non_empty_lines:
          string_without_empty_lines += line + "\n"
    text=string_without_empty_lines #This conatins no empty lines

    total_amount=total_amount.strip()

    # regex to match the first word in the line where total amount occurs and the first of the next line of total
    # amount. These are the start and the end parameters respectively
    start_end=re.search('\n.*?'+f"\s{total_amount}"+'.*?\n\w+', text)
    if(start_end!=None):
        start=start_end.group().split(" ")[0].strip()
        end=start_end.group().split("\n")[-1].strip()

    else:
        # If no regex matches
        print("start, end cannot be found")
    return "-1", "-1"

#function to get the desired fields when a template has been matched for the pdf
def getinf(file):

    global print_date
    global print_number
    global print_ammount
    global print_buyer
    global print_seller

    filename=os.path.join(os.getcwd(), "Invoices_info")
    filename=os.path.join(filename, file)
    dataframe1 = pd.read_excel(filename)
    key=file.split(".")[0]
    #Cropping and saving the raectangle that contains the keyword
    img1 = Image.open("page0.jpg")
    img1 = img1.crop((int(dataframe1['x1'].iloc[0])+shiftx,int(dataframe1['y1'].iloc[0]) +shifty ,
                      int(dataframe1['x2'].iloc[0])+shiftw,int(dataframe1['y2'].iloc[0])+shifth))
    # img1 = img1.crop((item["keyword_cordinates"]["x1"]+shiftx, item["keyword_cordinates"]
    #                  ["y1"]+shifty, item["keyword_cordinates"]["x2"]+shiftw, item["keyword_cordinates"]["y2"]+shifth))
    image=cv2.imread('page0.jpg')
    img1.save('img2.png')
    img1.close()
    #reading text from the cropped image to get the keyword
    text = str(image_to_string(
        Image.open(r"img2.png"), lang='eng'))


    #Cropping and saving the raectangle that contains the Date of Invoice
    img1 = Image.open("page0.jpg")
    img1 = img1.crop((int(dataframe1['x1'].iloc[1])+shiftx,int(dataframe1['y1'].iloc[1]) +shifty ,
                      int(dataframe1['x2'].iloc[1])+shiftw,int(dataframe1['y2'].iloc[1])+shifth))
    # img1 = img1.crop((item["Date"]["x1"]+shiftx, item["Date"]
    #                  ["y1"]+shifty, item["Date"]["x2"]+shiftw, item["Date"]["y2"]+shifth))
    img1.save('img2.png')
    img1.close()
    #reading text from the cropped image to get the Date of Invoice
    print_date = str(image_to_string(
        Image.open(r"img2.png"), lang='eng'))
    print("Date of Invoice: ", print_date)

    #reading the Invoice No after selecting the bounding box
    img1 = Image.open("page0.jpg")
    img1 = img1.crop((int(dataframe1['x1'].iloc[2])+shiftx,int(dataframe1['y1'].iloc[2]) +shifty ,
                      int(dataframe1['x2'].iloc[2])+shiftw,int(dataframe1['y2'].iloc[2])+shifth))
    # img1 = img1.crop((item["Invoice_No"]["x1"]+shiftx, item["Invoice_No"]
    #                  ["y1"]+shifty, item["Invoice_No"]["x2"]+shiftw, item["Invoice_No"]["y2"]+shifth))
    img1.save('img2.png')
    img1.close()
    print_number = str(image_to_string(
        Image.open(r"img2.png"), lang='eng'))
    print("invoice No ", print_number)

    #reading the Total bill after selecting the bounding box
    img1 = Image.open("page0.jpg")
    img1 = img1.crop((int(dataframe1['x1'].iloc[3])+shiftx,int(dataframe1['y1'].iloc[3]) +shifty ,
                      int(dataframe1['x2'].iloc[3])+shiftw,int(dataframe1['y2'].iloc[3])+shifth))
    # img1 = img1.crop((item["Total Bill"]["x1"]+shiftx, item["Total Bill"]
    #                  ["y1"]+shifty, item["Total Bill"]["x2"]+shiftw, item["Total Bill"]["y2"]+shifth))
    img1.save('img2.png')
    img1.close()
    print_ammount = str(image_to_string(
        Image.open(r"img2.png"), lang='eng'))
    print("Total Bill: ", print_ammount)


    #reading the Buyer Address after selecting the bounding box
    total_amount=text
    img1 = Image.open("page0.jpg")
    img1 = img1.crop((int(dataframe1['x1'].iloc[4])+shiftx,int(dataframe1['y1'].iloc[4]) +shifty ,
                      int(dataframe1['x2'].iloc[4])+shiftw,int(dataframe1['y2'].iloc[4])+shifth))
    # img1 = img1.crop((item["Buyer"]["x1"]+shiftx, item["Buyer"]
    #                  ["y1"]+shifty, item["Buyer"]["x2"]+shiftw,item["Buyer"]["y2"]+shifth))
    img1.save('img2.png')
    img1.close()
    print_buyer = str(image_to_string(
        Image.open(r"img2.png"), lang='eng'))
    print("Buyer: ", print_buyer)


    #reading the Seller Address  after selecting the bounding box
    img1 = Image.open("page0.jpg")
    img1 = img1.crop((int(dataframe1['x1'].iloc[5])+shiftx,int(dataframe1['y1'].iloc[5]) +shifty ,
                      int(dataframe1['x2'].iloc[5])+shiftw,int(dataframe1['y2'].iloc[5])+shifth))
    # img1 = img1.crop((item["Seller"]["x1"]+shiftx, item["Seller"]
    #                  ["y1"]+shifty, item["Seller"]["x2"]+shiftw, item["Seller"]["y2"]+shifth))
    img1.save('img2.png')
    img1.close()
    print_seller = str(image_to_string(
        Image.open(r"img2.png"), lang='eng'))
    print("Seller: ", print_seller)

    return total_amount


#Function for selecting the rectange by dragging the mouse
def shape_selection(event, x, y, flags, param):
    # grab references to the global variables

    global ref_point2, crop

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        # ref_point = [(x, y)]
        ref_point.append((x, y))
        # it+=1
        ref_point2 = [(x, y)]

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        if(curr == 1):
            ref_point.append((x, y))
            cv2.rectangle(image, ref_point[len(
                ref_point)-1], ref_point[len(ref_point)-2], (0, 255, 0), 2)
        elif(curr == 2):
            ref_point2.append((x, y))
            cv2.rectangle(image, ref_point2[0], ref_point2[1], (0, 255, 0), 2)
        # cv2.resizeWindow("image", 1000,1500)
        cv2.imshow("image", image)

#Connection to db
# connection = pymongo.MongoClient('127.0.0.1:27017')
# db = connection.invoiceocr


# Workbook is created
wb = Workbook()

# add_sheet is used to create sheet.
sheet1 = wb.add_sheet('Sheet 1')

sheet1.write( 0, 1, 'x1')
sheet1.write( 0, 2, 'y1')
sheet1.write( 0, 3,'x2')
sheet1.write( 0, 4,'y2')
sheet1.write( 1, 0,'keyword_cordinates')
sheet1.write( 2, 0,'Date')
sheet1.write( 3, 0,'Invoice_No')
sheet1.write( 4, 0,'Total Bill')
sheet1.write( 5, 0,'Buyer')
sheet1.write( 6, 0,'Seller')
sheet1.write( 7, 0, 'match_start')
sheet1.write( 8, 0, 'match_end')
sheet1.write( 9, 0, 'seller_key')
sheet1.write( 10, 0, 'no_of_pages')
#Intializing Some variables for part3
ref_point = []
total_amount=""
crop = False
onboard=0
flag=0
shiftx=0
shifty=0
shiftw=0
shifth=0
found=0
no_of_pages=0



st.title("Data Extraction from Invoices")
invoice_pdf = st.file_uploader("Upload a invoice")

if invoice_pdf:
    #converting pdf into images for every page of the pdf
    images = convert_from_path(invoice_pdf.name, poppler_path = './poppler-0.68.0/bin/')

    #Extracting the image of each pages from the pdf
    no_of_pages=len(images)
    for i in range(len(images)):
        images[i].save('page' + str(i) + '.jpg', 'JPEG')

    found = 0 #variable to check if a matching template is found
    image = cv2.imread('page0.jpg')
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)  # determine the skew angle prsent in the original image
    rotated = rotate(image, angle, (0, 0, 0))  #cancelling the skew in the original image and rorated is the new image after cancelling the skew in the original image
    cv2.imwrite('page0.jpg', rotated)
    matched_doc="" # to store the temaplate that has match with the template

    image = Image.open('page0.jpg')
    if st.button("View Invoice"):
        # st.image('page0.jpg')
        st.image(image)

    print("Select boxes in the order:\n1.Keyword\n2.Date of Invoice\n3.Invoice No.\n4.Total Bill Amount\n5.Buyer Details\n6.Seller Details   ")

    #extarcting the text of the page1
    text = str(image_to_string(Image.open(r"page0.jpg"), lang='eng'))

    #check if a matching template exists with teh same keyword and keyword bounding boxes
    for file in os.listdir(os.path.join(os.getcwd(), "Invoices_info")):
    # for document in db.invoices.find():
        img1 = Image.open("page0.jpg")
        filename=os.path.join(os.getcwd(), "Invoices_info")
        filename=os.path.join(filename, file)
        print(filename)
        dataframe1 = pd.read_excel(filename)

        img3 = img1.crop((int(dataframe1['x1'].iloc[0]),int(dataframe1['y1'].iloc[0]) ,
                          int(dataframe1['x2'].iloc[0]),int(dataframe1['y2'].iloc[0])))
        # img3 = img1.crop((document["keyword_cordinates"]["x1"], document["keyword_cordinates"]
        #                  ["y1"], document["keyword_cordinates"]["x2"], document["keyword_cordinates"]["y2"]))
        img3.save('img3.png')
        img3.close()
        image = Image.open('img3.png')
        image.close()
        text = str(image_to_string(
            Image.open(r"img3.png"), lang='eng'))
        text=text.strip()
        foundregex=re.search(r'[a-zA-Z]+', text)
        if(foundregex!=None):
            text=foundregex.group()
        # key=document["keyword"].strip()
        key=file.split(".")[0]
        if(text== key):
            found = 1
            total_amount=getinf(file)
            matched_doc=key
            if(no_of_pages!=dataframe1['x1'].iloc[9]):
                print("-1")
                flag=-1
                break

    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # to make tesseract work, put in the exact path of tesseract

    #matching every word of the pdf to every keyword to find the shift and confirming by matching with seller key
    if found==0:
        for file in os.listdir(os.path.join(os.getcwd(), "Invoices_info")):
        # for document in db.invoices.find():
            img = Image.open('page0.jpg')
            data = image_to_data(img, output_type='dict')
            boxes = len(data['level'])
            filename=os.path.join(os.getcwd(), "Invoices_info")
            filename=os.path.join(filename, file)
            dataframe1 = pd.read_excel(filename)
            key=file.split(".")[0]

            for i in range(boxes):

                if data['text'][i].strip()!= ''.strip():
                    key_to_match=data['text'][i]
                    foundregex=re.search(r'[a-zA-Z]+', key_to_match)

                    #Checking only for valid strings
                    if(foundregex!=None):
                        key_to_match=foundregex.group()

                    #If keyword matches with a word in the pdf, get the x, y shift of teh keyword coordinates stored

                    if((key_to_match)==key):

                        shiftx=data["left"][i]-dataframe1["x1"].iloc[0]
                        shifty=data["top"][i]- dataframe1["y1"].iloc[0]
                        shiftw=data["left"][i]-dataframe1["x1"].iloc[0]
                        shifth=data["top"][i]- dataframe1["y1"].iloc[0]
                        img1 = Image.open("page0.jpg")
                        # print(dataframe1["x1"].iloc[5]+shiftx,"  ", int(dataframe1["x1"].iloc[5]) +shiftx," ", int(dataframe1["x2"].iloc[5])+shifty," ", int(dataframe1["y1"].iloc[5])+shiftw, " ", int(dataframe1["y2"].iloc[5])+shifth)
                        img1 = img1.crop((( int(dataframe1["x1"].iloc[5]) +shiftx, int(dataframe1["y1"].iloc[5])+shifty, int(dataframe1["x2"].iloc[5])+shiftw, int(dataframe1["y2"].iloc[5])+shifth)) )

                        # img1 = img1.crop((document["Seller"]["x1"]+shiftx, document["Seller"]
                        #                  ["y1"]+shifty, document["Seller"]["x2"]+shiftw, document["Seller"]["y2"]+shifth))
                        img1.save('img2.png')
                        img1.close()
                        text = str(image_to_string(Image.open(r"img2.png"), lang='eng'))

                        #Confirming if the seller key matches then only we consider this template otherwise we send it for onboarding
                        if( text.split("\n")[0].strip()==dataframe1["x1"].iloc[8].strip()):
                            found=1
                            break


            if found==1:
                matched_doc=key
                total_amount=getinf(file)
                if(no_of_pages!=dataframe1['x1'].iloc[9]):
                    print("-1")
                    flag=-1 #flag indicates if nof pages in teh onboarded pdf and teh current pdf are same or not, if not sent for manual (-1)
                    break

                break

    #When no template has been matched
    if found == 0 and flag!=-1:
        onboard=1
        st.header("No Template found for this Invoice; \nPlease Start the onboarding\n")
        if st.button("Onboard"):

            image = cv2.imread("page0.jpg")
            clone = image.copy()

            cv2.namedWindow("image", cv2.WINDOW_NORMAL) #resizing the output window
            cv2.setMouseCallback("image", shape_selection) #to select rectangle by pressing and releasing mousebutton
            curr = 1
            while True:
                cv2.imshow("image", image)
                key = cv2.waitKey(1) & 0xFF

                #When c is prssed on the keyword the opencv window closes
                if key == ord("c"):
                    break

            #If valid selection is made by user while dragging the rectangle, i.e, atleast one rectangle has been selected successfully
            if len(ref_point) >= 2:
                img1 = Image.open("page0.jpg")
                #cropping and saving only the rectangle portion of the image frmo where we have to extract the text
                img3 = img1.crop(
                    (ref_point[0][0], ref_point[0][1], ref_point[1][0], ref_point[1][1]))
                img3.save('img2.png')
                img3.close()

                image = Image.open('img2.png')
                image.close()

                #text variable contains the text in the bounding box selected
                text = str(image_to_string(
                    Image.open(r"img2.png"), lang='eng'))

                #getting the x and y coordinates of the keyword  from the pdf, for this the text read from the bounding box is matxhed with edvery word
                #of the pdf and when the word matches we store its x, y coordinates as the keyword's coordinates. This is done to ensure that we dont consider the extra region of the image that ahs not
                # text , we want a tight bound to x y coordinates for keyword
                text=text.strip()
                foundregex=re.search(r'[a-zA-Z]+', text)
                if(foundregex!=None):
                    text=foundregex.group()

                #onverting the whole pdf to text to match its every word
                myimg=Image.open('page0.jpg')
                data = image_to_data(myimg, output_type='dict')
                boxes = len(data['level'])

                for i in range(boxes):
                    key_to_match=data['text'][i].strip()
                    if(key_to_match!=""):
                        foundregex=re.search(r'[a-zA-Z]+', key_to_match)
                        if(foundregex!=None):
                            key_to_match=foundregex.group()

                        if key_to_match.strip() == text.strip():
                            break

                #inserting the new template in the db
                # db.invoices.insert_one({"keyword":text})


                keyword = text
                name=keyword+'.xls'
                # wb.save(name)
                # db.invoices.update_one({"keyword": keyword}, {"$set": {"no_of_pages": no_of_pages}})
                sheet1.write(10, 1, no_of_pages)
                # db.invoices.update_one({"keyword": keyword}, {
                # "$set": {"keyword_cordinates": {"x1":data["left"][i], "y1":data["top"][i], "x2": data["left"][i]+data["width"][i], "y2":data["top"][i]+data["height"][i]}}})
                sheet1.write(1, 1, data["left"][i])
                sheet1.write(1, 2, data["top"][i])
                sheet1.write(1, 3, data["left"][i]+data["width"][i])
                sheet1.write(1, 4, data["top"][i]+data["height"][i])

                #extracting the date of invoice from the bounding box selected for it and storing its keyword coordinates in the database
                img3 = img1.crop(
                (ref_point[2][0], ref_point[2][1], ref_point[3][0], ref_point[3][1]))
                # db.invoices.update_one({"keyword": keyword}, {
                # "$set": {"Date": {"x1": ref_point[2][0], "y1": ref_point[2][1], "x2": ref_point[3][0], "y2": ref_point[3][1]}}})
                sheet1.write(2, 1, ref_point[2][0])
                sheet1.write(2, 2, ref_point[2][1])
                sheet1.write(2, 3,  ref_point[3][0])
                sheet1.write(2, 4,  ref_point[3][1])
                img3.save('img2.png')
                img3.close()
                image = Image.open('img2.png')
                image.close()
                text = str(image_to_string(Image.open(r"img2.png"), lang='eng'))
                print_date=text.strip()
                st.write("Invoice Date",print_date)
                #extracting the Invoice No. from the bounding box selected for it and storing its keyword coordinates in the database
                img3 = img1.crop((ref_point[4][0], ref_point[4][1], ref_point[5][0], ref_point[5][1]))
                # db.invoices.update_one({"keyword": keyword}, {
                # "$set": {"Invoice_No": {"x1": ref_point[4][0], "y1": ref_point[4][1], "x2": ref_point[5][0], "y2": ref_point[5][1]}}})
                sheet1.write(3, 1, ref_point[4][0])
                sheet1.write(3, 2, ref_point[4][1])
                sheet1.write(3, 3,  ref_point[5][0])
                sheet1.write(3, 4,  ref_point[5][1])
                img3.save('img2.png')
                img3.close()
                image = Image.open('img2.png')
                image.close()
                text = str(image_to_string(Image.open(r"img2.png"), lang='eng'))
                print_number=text.strip()
                print("Invoice No: ", print_number)

                #extracting the Total Bill from the bounding box selected for it and storing its keyword coordinates in the database
                img3 = img1.crop(
                (ref_point[6][0], ref_point[6][1], ref_point[7][0], ref_point[7][1]))
                # db.invoices.update_one({"keyword": keyword}, {
                # "$set": {"Total Bill": {"x1": ref_point[6][0], "y1": ref_point[6][1], "x2": ref_point[7][0], "y2": ref_point[7][1]}}})
                sheet1.write(4, 1, ref_point[6][0])
                sheet1.write(4, 2, ref_point[6][1])
                sheet1.write(4, 3,  ref_point[7][0])
                sheet1.write(4, 4,  ref_point[7][1])
                img3.save('img2.png')
                img3.close()
                image = Image.open('img2.png')
                image.close()
                text = str(image_to_string(
                Image.open(r"img2.png"), lang='eng'))
                print_ammount=text.strip()
                print("Total Bill ", print_ammount)
                total_amount=text

                #Stroring the word before and after the total bill for part 3
                wholetext = str(image_to_string(
                    Image.open(r"page0.jpg"), lang='eng'))
                wholetext=wholetext.strip()
                match_start, match_end=getstartend(total_amount, wholetext)

                # db.invoices.update_one({"keyword": keyword}, {
                #     "$set": {"match_start": match_start}})
                sheet1.write(7,1,match_start)
                # db.invoices.update_one({"keyword": keyword}, {
                #     "$set": {"match_end": match_end}})
                sheet1.write(8,1,match_end)

                #extracting the Buyer Address from the bounding box selected for it and storing its keyword coordinates in the database
                img3 = img1.crop(
                (ref_point[8][0], ref_point[8][1], ref_point[9][0], ref_point[9][1]))
                # db.invoices.update_one({"keyword": keyword}, {
                # "$set": {"Buyer": {"x1": ref_point[8][0], "y1": ref_point[8][1], "x2": ref_point[9][0], "y2": ref_point[9][1]}}})
                sheet1.write(5, 1, ref_point[8][0])
                sheet1.write(5, 2, ref_point[8][1])
                sheet1.write(5, 3,  ref_point[9][0])
                sheet1.write(5, 4,  ref_point[9][1])
                img3.save('img2.png')
                img3.close()
                image = Image.open('img2.png')
                image.close()
                text = str(image_to_string(
                Image.open(r"img2.png"), lang='eng'))
                print_buyer=text.strip()
                print("Buyer: ", print_buyer)
                #extracting the Seller Address from the bounding box selected for it and storing its keyword coordinates in the database
                img3 = img1.crop(
                (ref_point[10][0], ref_point[10][1], ref_point[11][0], ref_point[11][1]))
                # db.invoices.update_one({"keyword": keyword}, {
                # "$set": {"Seller": {"x1": ref_point[10][0], "y1": ref_point[10][1], "x2": ref_point[11][0], "y2": ref_point[11][1]}}})
                sheet1.write(6, 1, ref_point[10][0])
                sheet1.write(6, 2, ref_point[10][1])
                sheet1.write(6, 3,  ref_point[11][0])
                sheet1.write(6, 4,  ref_point[11][1])
                img3.save('img2.png')
                img3.close()
                image = Image.open('img2.png')
                image.close()
                text = str(image_to_string(
                Image.open(r"img2.png"), lang='eng'))
                print_seller=text.strip()
                sellerkey=print_seller.split("\n")[0]
                # db.invoices.update_one({"keyword": keyword},{"$set": {"seller_key": sellerkey}})
                sheet1.write(9,1,sellerkey)
                print("Seller: ", print_seller)
                wb.save(name)
                dst_path=os.path.join(os.getcwd(), "Invoices_info")
                dst_path=os.path.join(dst_path, name)
                src_path=os.path.join(os.getcwd(), name)
                shutil.move(src_path, dst_path)

            cv2.destroyAllWindows()


    d = {
                "Invoice Date:": print_date,
                "Invoice Number:": print_number,
                "Invoice Amount": print_ammount,
                "Invoice Buyer": print_buyer,
                "Invoice Seller": print_seller
            }

    print(d)

    if st.button("Extract Data\n"):
        st.write(d)
