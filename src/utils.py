import os
import shutil
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from pytube import YouTube
from decord import AudioReader
from scipy.io.wavfile import write
from pyAudioAnalysis.audioBasicIO import read_audio_file
from pyAudioAnalysis import MidTermFeatures as mT
from scipy.spatial import distance


'''
'''
def ytDownloader(link,videoPath=None,audioPath=None,mode="app"):
    
    try:
        yt = YouTube(link)
    except:
        print("Cant connect to download: {}".format(link))
        print(30*"-")

    if mode == "app":
        videoPath = "temp/videoTest.mp4"
        audioPath = "temp/audioTest.wav"
    else:
        print("[INFO]       Downloading: {}".format(link))
    
    if not os.path.isfile(videoPath) or not os.path.isfile(audioPath):
        videoFilter = yt.streams.filter(file_extension='mp4')
        stream = yt.streams.get_by_itag(18)

        stream.download(os.path.split(videoPath)[0],os.path.split(videoPath)[1])

        fs = 44100

        ar = AudioReader(videoPath, sample_rate=fs, mono=True)
        ar = ar[0:ar.shape[1]]
        ar = ar.asnumpy()

        audioData = np.int16(ar/np.max(np.abs(ar)) * 32767)
        write(audioPath,fs,audioData.T)

    return yt.title, yt.embed_url, yt.thumbnail_url

'''
'''
def embedUrlFromLink(link):
    try:
        yt = YouTube(link)
    except:
        print("Cant connect to Youtube")
        print(30*"-")

    return yt.embed_url

'''
'''
def extractHCAudioFeatures(audioFile):

    fs, signal = read_audio_file(audioFile)

    HCAudioFeatures, _, _ = mT.mid_feature_extraction(signal, fs, 
                                                1 * fs,
                                                1 * fs,
                                                0.05 * fs,
                                                0.05 * fs)

    HCAudioFeatures = np.median(HCAudioFeatures,axis=1)

    return HCAudioFeatures

'''
'''
def extractDeepVideoFeatures(videoPath):
    print("[INFO]       Deep Video Features Extraction...")

    os.makedirs("tmpKF",exist_ok=True)
    # https://stackoverflow.com/questions/9064962/key-frame-extraction-from-video
    os.system("ffmpeg -i {} -vf select='eq(pict_type\,PICT_TYPE_I) -vsync 2 -s 224x224 -f image2 tmpKF/thumbnails-%02d.png".format(videoPath))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    resnet = models.resnet152(pretrained=True)
    modules = list(resnet.children())[:-1]
    resnet = nn.Sequential(*modules)
    resnet = resnet.to(device)

    transformations =  transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize([224,224]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    spatialFts = list()

    with torch.no_grad():
        for th in sorted(os.listdir("tmpKF")):
            kf = cv2.imread(os.path.join("tmpKF",th))
            kf = cv2.cvtColor(kf, cv2.COLOR_BGR2RGB)

            frame = transformations(kf)
            frame = frame.reshape((1,frame.shape[0],frame.shape[1],frame.shape[2]))
            frame = frame.to(device)
            x = resnet(frame)
            x = x.view(x.size(0), -1)

            spatialFts.append(x)
            
        spatialFts = torch.stack(spatialFts, dim=0).transpose_(0, 1)
        spatialFts = spatialFts[0,:,:]
        meanSpatialFts = torch.mean(spatialFts,0)
        stdSpatialFts = torch.std(spatialFts,0)

        if os.path.isdir("tmpKF"):
            shutil.rmtree("tmpKF")

        return meanSpatialFts.cpu().detach().numpy(), stdSpatialFts.cpu().detach().numpy()

'''
'''
def audioBasedRetrieval(link,vcBase,dist="chebyshev"):
    dbFeatures = list()

    for _, info in vcBase.MVBaseInfoFile.items():
        audioF = np.load(info["FeaturesPath"])["HCAudioFeatures"]
        dbFeatures.append(audioF)

    dbFeatures = np.array(dbFeatures)

    tmpDir = "temp"
    os.makedirs(tmpDir,exist_ok=True)
    ytDownloader(link)

    testAudioF = extractHCAudioFeatures("temp/audioTest.wav")
    testAudioF = testAudioF.reshape(1,-1)

    distMatrix = distance.cdist(testAudioF,dbFeatures,dist)
    retrieved = np.argpartition(distMatrix[0], 5)[:5]    
    
    if os.path.isdir(tmpDir):
        shutil.rmtree(tmpDir)
    
    return retrieved

'''
'''
def videoBasedRetrieval(link,vcBase,dist="euclidean"):
    dbFeatures = list()

    for _, info in vcBase.MVBaseInfoFile.items():
        meanVideoF = np.load(info["FeaturesPath"])["meanDeepVideoFeatures"]
        stdVideoF = np.load(info["FeaturesPath"])["stdDeepVideoFeatures"]
        videoF =  np.concatenate([meanVideoF,stdVideoF])
        dbFeatures.append(videoF)

    dbFeatures = np.array(dbFeatures)

    tmpDir = "temp"
    os.makedirs(tmpDir,exist_ok=True)
    ytDownloader(link)   
                    
    meanVideoF, stdVideoF = extractDeepVideoFeatures("temp/videoTest.mp4")
    testVideoF =  np.concatenate([meanVideoF,stdVideoF])
    testVideoF = testVideoF.reshape(1,-1)

    distMatrix = distance.cdist(testVideoF,dbFeatures,dist)
    retrieved = np.argpartition(distMatrix[0], 5)[:5]
    
    if os.path.isdir(tmpDir):
        shutil.rmtree(tmpDir)

    return retrieved

'''
'''
def multimodalRetrieval(link,vcBase,dist="braycurtis"):
    dbFeatures = list()

    for idx, info in vcBase.MVBaseInfoFile.items():
        audioF = np.load(info["FeaturesPath"])["HCAudioFeatures"]
        meanVideoF = np.load(info["FeaturesPath"])["meanDeepVideoFeatures"]
        stdVideoF = np.load(info["FeaturesPath"])["stdDeepVideoFeatures"]
        feats = np.concatenate([meanVideoF,stdVideoF,audioF])
        
        dbFeatures.append(feats)
    
    dbFeatures = np.array(dbFeatures)

    tmpDir = "temp"
    os.makedirs(tmpDir,exist_ok=True)
    ytDownloader(link)

    testAudioF = extractHCAudioFeatures("temp/audioTest.wav")
    meanVideoF, stdVideoF = extractDeepVideoFeatures("temp/videoTest.mp4")
    testF = np.concatenate([meanVideoF,stdVideoF,testAudioF])

    testF = testF.reshape(1,-1)

    distMatrix = distance.cdist(testF,dbFeatures,dist)
    retrieved = np.argpartition(distMatrix[0], 5)[:5] 
    
    if os.path.isdir(tmpDir):
        shutil.rmtree(tmpDir)

    return retrieved