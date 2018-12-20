from xml.dom import minidom
import os
import argparse



if __name__ == '__main__':
  # Argparse
    parser = argparse.ArgumentParser(description='Database generator for UA-DETRAC dataset')
    parser.add_argument('--img_dir', help='the dir containing dataset imgs')
    parser.add_argument('--query_dir', help='the dir containing dataset imgs')
    parser.add_argument('--gallery_dir', help='the dir containing dataset imgs')
    parser.add_argument('--label_dir', help='the dir containing dataset label files (MAT format)')
    parser.add_argument('--train_txt', help='the output txt file listing all imgs to database and its label')
    parser.add_argument('--query_txt', help='the output txt file listing all imgs to database and its label')
    parser.add_argument('--gallery_txt', help='the output txt file listing all imgs to database and its label')
    args = parser.parse_args()


    xmlfile = open(args.label_dir,'r')
    xmldoc = xmldoc = minidom.parse(xmlfile)
    itemlist = xmldoc.getElementsByTagName('Item')

    img_dir = args.img_dir

    txt_file = open(args.train_txt,'w')
    
    img_list = []
    V_ID_list = []
    colorID_list = []
    typeID_list = []
    V_ID_dict = {}
    count = 1
    for s in itemlist:
        img_name = s.attributes['imageName'].value
        img_list.append(os.path.join(img_dir,img_name))

        V_ID = s.attributes['vehicleID'].value
        if V_ID not in V_ID_dict:
            V_ID_dict[V_ID]=count
            count +=1 
        V_ID_list.append(V_ID)
        colorID_list.append(s.attributes['colorID'].value)
        typeID_list.append(s.attributes['typeID'].value)

    txt_file.write('img_path id color type\n')
    for i in range(len(img_list)):
        img_path = img_list[i]
        V_ID = str(V_ID_dict[V_ID_list[i]])
        colorID = colorID_list[i]
        typeID = typeID_list[i]
        txt_file.write(img_path+' '+V_ID+' '+colorID+' '+typeID+'\n')
    txt_file.close()
    xmlfile.close()

    #query 
    img_list = os.listdir(args.query_dir)
    img_list.sort()
    file = open(args.query_txt,'w')
    file.write('img_path\n')
    for i in img_list:
        file.write(os.path.join(args.query_dir,i)+'\n')
    file.close()
    #gallery
    #query 
    img_list = os.listdir(args.gallery_dir)
    img_list.sort()
    file = open(args.gallery_txt,'w')
    file.write('img_path\n')
    for i in img_list:
        file.write(os.path.join(args.gallery_dir,i)+'\n')
    file.close()



