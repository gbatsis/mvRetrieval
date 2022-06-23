import os
import json
import pandas as pd
import plotly.express as px

from scipy.spatial import distance
from utils import *

class VCStorage():
    def __init__(self):
        self.initFile = "initFile.json"
        self.baseDir = os.path.join("E:\\AI\\MULTIMODAL","MusicVideoBase")
        self.MVInfoFileName = os.path.join(self.baseDir,"MVInfo.json")
        self.MVBaseInfoFileName = os.path.join(self.baseDir,"MVBaseInfo.json")
        self.videoDir = os.path.join(self.baseDir,"Video")
        self.audioDir = os.path.join(self.baseDir,"Audio")
        self.featuresDir = os.path.join(self.baseDir,"Features")
        self.MVInfoFile = self.getMVInfo()
        self.MVBaseInfoFile = self.getMVBaseInfo()
        
    '''
    '''
    def getMVInfo(self):
        if not os.path.isfile(self.MVInfoFileName):
            MVInfoFile = self.receiveVideos()
            
            with open(self.MVInfoFileName, 'w', encoding='utf8') as f:
                json.dump(MVInfoFile,f,indent=4,ensure_ascii=False)
        else:
            return json.load(open(self.MVInfoFileName))

    '''
    '''
    def receiveVideos(self):
        print("[INFO]   Generating Music Video Database.")

        os.makedirs(self.baseDir,exist_ok=True)
        os.makedirs(self.videoDir,exist_ok=True)
        os.makedirs(self.audioDir,exist_ok=True)

        initFile = json.load(open(self.initFile))
        
        baseIdx = 0

        MVInfo = dict()

        for genre, videoList in initFile.items():
            for idx,vid in enumerate(videoList):
                videoFileName = "{}_{}.mp4".format(genre,idx)
                audioFileName = "{}_{}.wav".format(genre,idx)

                videoPath = os.path.join(self.videoDir,videoFileName)
                audioPath = os.path.join(self.audioDir,audioFileName)
                
                ytTitle, embedUrl, thUrl = ytDownloader(vid,videoPath,audioPath,"db")

                vcInfo = {
                    "ytTitle" : ytTitle,
                    "ytLink" : vid,
                    "embedUrl" : embedUrl, 
                    "thUrl" : thUrl,
                    "Genre" : genre,
                    "VideoPath" : videoPath,
                    "AudioPath" : audioPath
                }

                MVInfo[baseIdx] = vcInfo

                baseIdx += 1

        return MVInfo

    '''
    '''
    def getMVBaseInfo(self):
        if not os.path.isfile(self.MVBaseInfoFileName):
            VCBaseInfoFile = self.extractMVBaseInfo()
            with open(self.MVBaseInfoFileName, 'w', encoding='utf8') as f:
                json.dump(VCBaseInfoFile,f,indent=4,ensure_ascii=False)
        else:
            return json.load(open(self.MVBaseInfoFileName))

    '''
    '''
    def extractMVBaseInfo(self):
        print("[INFO]   Extracting Video Database Information.")

        os.makedirs(self.featuresDir,exist_ok=True)
        MVBaseInfo = dict()

        for vcIdx, info in self.MVInfoFile.items():
            vc = info
            featuresPath = os.path.join(self.featuresDir,"Features_{}.npz".format(vcIdx))
            if not os.path.isfile(featuresPath):
                HCAudioFeatures = extractHCAudioFeatures(info["AudioPath"])
                meanDeepVideoFeatures,stdDeepVideoFeatures = extractDeepVideoFeatures(info["VideoPath"])
                np.savez(featuresPath, HCAudioFeatures=HCAudioFeatures, meanDeepVideoFeatures=meanDeepVideoFeatures, stdDeepVideoFeatures=stdDeepVideoFeatures)
            vc["FeaturesPath"] = featuresPath
            MVBaseInfo[vcIdx] = vc
        return MVBaseInfo

'''
'''
def generateValidationData():
    valVids = [("HipHop","https://www.youtube.com/watch?v=i9gruD-5dnw")
    ,("HipHop","https://www.youtube.com/watch?v=dDO2j8eNHos") 
    ,("HipHop","https://www.youtube.com/watch?v=viGMOD4NXNk") 
    ,("HipHop","https://www.youtube.com/watch?v=TpGi48j17_8") 
    ,("HipHop","https://www.youtube.com/watch?v=EK3zVJbuVhY") 
    ,("Pop","https://www.youtube.com/watch?v=nfWlot6h_JM")
    ,("Pop","https://www.youtube.com/watch?v=vMLk_T0PPbk")
    ,("Pop","https://www.youtube.com/watch?v=L0X03zR0rQk")
    ,("Pop","https://www.youtube.com/watch?v=tcYodQoapMg")
    ,("Pop","https://www.youtube.com/watch?v=Nq3x1AkwgpY")]

    valDataDir = "valData"
    os.makedirs(valDataDir,exist_ok=True)

    for idx, vc in enumerate(valVids):
        tmpDir = "temp"
        os.makedirs(tmpDir,exist_ok=True)

        ytDownloader(vc[1])

        testAudioF = extractHCAudioFeatures("temp/audioTest.wav")
        testVideoFmean, testVideoFstd = extractDeepVideoFeatures("temp/videoTest.mp4")
    
        np.savez(os.path.join(valDataDir,"features_{}_{}".format(vc[0],idx)), HCAudioFeatures=testAudioF,meanDeepVideoFeatures=testVideoFmean,stdDeepVideoFeatures=testVideoFstd)
                    
        if os.path.isdir(tmpDir):
            shutil.rmtree(tmpDir)

'''
'''
def validation(vcBase):
    modalities = ["audio","video","multimodal"]
    distanceMetrics = [
                    "euclidean","cityblock","cosine","correlation",
                    "hamming","jaccard","jensenshannon","chebyshev",
                    "canberra","braycurtis","sokalsneath"]
    valDataDir = "valData"

    results = list()

    for m in modalities:
        print("[INFO] Modality: {}".format(m))
        
        dbFeatures = list()
        for _, info in vcBase.MVBaseInfoFile.items():
            if m == "audio":
                audioF = np.load(info["FeaturesPath"])["HCAudioFeatures"]
                dbFeatures.append(audioF)
            elif m == "video":
                meanVideoF = np.load(info["FeaturesPath"])["meanDeepVideoFeatures"]
                stdVideoF = np.load(info["FeaturesPath"])["stdDeepVideoFeatures"]
                videoF =  np.concatenate([meanVideoF,stdVideoF])
                dbFeatures.append(videoF)
            elif m == "multimodal":
                audioF = np.load(info["FeaturesPath"])["HCAudioFeatures"]
                meanVideoF = np.load(info["FeaturesPath"])["meanDeepVideoFeatures"]
                stdVideoF = np.load(info["FeaturesPath"])["stdDeepVideoFeatures"]
                feats = np.concatenate([meanVideoF,stdVideoF,audioF])
                dbFeatures.append(feats)

        dbFeatures = np.array(dbFeatures)

        for dist in distanceMetrics:
            print("[INFO]      Distance: {}".format(dist))

            cor = 0

            for testVid in os.listdir(valDataDir):
                featuresPath = os.path.join(valDataDir,testVid)
                if m == "audio":
                    testF = np.load(featuresPath)["HCAudioFeatures"]
                elif m == "video":
                    meanVideoF = np.load(featuresPath)["meanDeepVideoFeatures"]
                    stdVideoF = np.load(featuresPath)["stdDeepVideoFeatures"]
                    testF =  np.concatenate([meanVideoF,stdVideoF])     
                elif m == "multimodal":
                    audioF = np.load(featuresPath)["HCAudioFeatures"]
                    meanVideoF = np.load(featuresPath)["meanDeepVideoFeatures"]
                    stdVideoF = np.load(featuresPath)["stdDeepVideoFeatures"]
                    testF = np.concatenate([meanVideoF,stdVideoF,audioF])
                
                testF = testF.reshape(1,-1)

                distMatrix = distance.cdist(testF,dbFeatures,dist)

                retIdx = np.argpartition(distMatrix[0], 5)[:5]

                for idx in retIdx:
                    gen = testVid.split("_")[1]
                    if gen == vcBase.MVBaseInfoFile[str(idx)]["Genre"]:
                        cor += 1

            results.append({
                "modality":m,
                "distance":dist,
                "correctPredictions":cor
            })

            print("[INFO]           Correct Genre Predictions: {}/{}".format(cor,len(os.listdir(valDataDir))*5))
           

        
        print(30*"-")

    resultsDF = pd.DataFrame.from_dict(results)
    resultsDF.to_csv("Results.csv")

'''
'''
def plotResults(resultsDF):
    for m in np.unique(resultsDF["modality"]):
        
        modDF = resultsDF[resultsDF["modality"] == m]
        modDF.columns = ["modality","Distance","Correct Predictions"]
        fig = px.bar(modDF,x="Distance",y="Correct Predictions",color='Correct Predictions',title="Correct Predictions for modality: {} - MAX = 50".format(m),template="plotly_dark")
        fig.update(layout_showlegend=False)
        fig.update_coloraxes(showscale=False)

        fig.show()

'''
'''
def main():
    vcBase = VCStorage()

    if not os.path.isdir("valData"):
        generateValidationData()

    if not os.path.isfile("Results.csv"):
        validation(vcBase)

    resultsDF = pd.read_csv("Results.csv",index_col=0)
    plotResults(resultsDF)

if __name__ == "__main__":
    main()