#!/usr/bin/python3
import os

#camera = icubCamera
camera = "realsense"


def eye_contact_create_annotation_file():
    print("Creating annotations for eyecontact dataset...")
    count_eyecontact = 0
    count_no_eyecontact = 0

    output_file = open(dataset + '/eyecontact_annotations.txt', "w+")

    list_participant = [name for name in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, name))]
    list_participant.sort()

    for participant in list_participant:
        print("Processing participant: ", participant)
        folder_participant = os.path.join(dataset, participant)

        # annotation for eye-contact
        eyecontact_dir = os.path.join(folder_participant, 'eyecontact_images_human/eyecontact_condition')
        files = os.listdir(eyecontact_dir)
        img_files = list(filter(lambda x: '.jpg' in x, files))
        img_files.sort()
        count_eyecontact = count_eyecontact + len(img_files)
        for img in img_files:
            output_file.write("./%s/eyecontact_images_human/eyecontact_condition/%s 1\n" % (participant, img))

        # annotation for no eye-contact
        no_eyecontact_dir = os.path.join(folder_participant, 'eyecontact_images_human/no_eyecontact_condition')
        files = os.listdir(no_eyecontact_dir)
        img_files = list(filter(lambda x: '.jpg' in x, files))
        img_files.sort()
        count_no_eyecontact = count_no_eyecontact + len(img_files)
        for img in img_files:
            output_file.write("./%s/eyecontact_images_human/no_eyecontact_condition/%s 0\n" % (participant, img))

    output_file.close()
    print("Number of eyecontact samples: ", count_eyecontact)
    print("Number of no eyecontact samples: ", count_no_eyecontact)


# ---------------------------------------------------------------

root_dir = os.path.join(os.getcwd(), "..", "datasets-preprocessing")
dataset = os.path.join(root_dir, camera)

eye_contact_create_annotation_file()
