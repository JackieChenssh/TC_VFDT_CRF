class DataProcess:

    def __init__(self,spt_path = None):
        if(spt_path != None):
            from numpy import load
            self.spt_point = load(spt_path) # col,bins

    def datasetIO(self,file,mode = 'wb',variable = None):
        # Read or Save Variable 
        import numpy as np
        assert not(mode == 'wb' and variable == None),'Cannot save None as file.'
        assert mode in ('wb','rb'),'Parameter mode only accept \'wb\' or \'rb\'.'

        import pickle
        with open(file, mode) as f:
            if(mode == 'rb'):
                variable  = pickle.load(f, encoding = 'bytes')
            else:
                pickle.dump(variable,f,protocol = 2)            

        return variable

    def feature_histeq(self,img):
        import numpy as np
        imhist,bins = np.histogram(img.flatten(),256)
        cdf = imhist.cumsum() #cumulative distribution function
        return np.interp(img.flatten(),bins[:-1],255 * cdf / cdf[-1]).astype(img.dtype)

    def feature_eigenvector(self,img,return_mean = True,return_spt = True):
        import numpy as np
        eigenvalue,eigenvector = np.linalg.eig(img)
        feature = abs(eigenvalue * eigenvector)

        if return_spt:
            feature_labeled = np.zeros(feature.shape,dtype = np.uint8)
            for col in range(feature.shape[1]):
                for spt in self.spt_point[col]:
                    feature_labeled[feature[:,col] > spt,col] += 1 

        return feature,(np.mean(feature,axis = 0) if return_mean else None),(feature_labeled if return_spt else None)

    def getFeature(self,imgArray):
        # Generate feature for ONE image
        import numpy as np
        from tqdm.notebook import tqdm
    #         assert isinstance(imgArray,np.ndarray),'Input must be numpy.ndarray type.'
    #         assert len(imgArray.shape) == 3 or (len(imgArray.shape) == 3 and imgArray.shape[0] == 3),'Input must be Gray or seperated RGB'
        imgArray = np.array(imgArray)

        img_histeqs = []       
        img_nh_absvector     = [] # continuous,no histeq img
        img_nh_absvectormean = [] # continuous one vector for three layers,no histeq img
        img_nh_sptabsvector  = [] # discrete,no histeq img
        img_h_absvector      = [] # continuous,histeq img
        img_h_absvectormean  = [] # continuous one vector for three layers,histeq img
        img_h_sptabsvector   = [] # discrete,histeq img

        if(len(imgArray[0].shape) == 2):
            img_histeqs = np.array([self.feature_histeq(img) for img in imgArray]).reshape(imgArray.shape) # continuous can be translated as discrete
            for img,img_histeq in tqdm(list(zip(imgArray,img_histeqs))):
                nh_absvector,nh_absvectormean,nh_sptabsvector = self.feature_eigenvector(img)
                h_absvector,h_absvectormean,h_sptabsvector    = self.feature_eigenvector(img_histeq)
                img_nh_absvector.append(nh_absvector)
                img_nh_absvectormean.append(nh_absvectormean)
                img_nh_sptabsvector.append(nh_sptabsvector)
                img_h_absvector.append(h_absvector)
                img_h_absvectormean.append(h_absvectormean)
                img_h_sptabsvector.append(h_sptabsvector)
    #                 pdb.set_trace()
        elif(len(imgArray[0].shape) == 3 and imgArray[0].shape[0] == 3): # RGB splited color
            img_histeqs = np.array([self.feature_histeq(img) for img in imgArray]).reshape(imgArray.shape) # continuous can be translated as discrete
            for img,img_histeq in tqdm(list(zip(imgArray,img_histeqs))):
                for colorLayer,colorLayer_hq in zip(img,img_histeq):
                    nh_absvector,nh_absvectormean,nh_sptabsvector = self.feature_eigenvector(colorLayer)
                    h_absvector,h_absvectormean,h_sptabsvector    = self.feature_eigenvector(colorLayer_hq)
                    img_nh_absvector.append(nh_absvector)
                    img_nh_absvectormean.append(nh_absvectormean)
                    img_nh_sptabsvector.append(nh_sptabsvector)
                    img_h_absvector.append(h_absvector)
                    img_h_absvectormean.append(h_absvectormean)
                    img_h_sptabsvector.append(h_sptabsvector)
    #                     pdb.set_trace()
    #         pdb.set_trace()

        return (img_histeqs,
                np.array(img_nh_absvector).reshape(-1,img_nh_absvector[0].shape[-1]),
                np.array(img_nh_absvectormean).reshape(-1,img_nh_absvectormean[0].shape[-1]),
                np.array(img_nh_sptabsvector).reshape(-1,img_nh_sptabsvector[0].shape[-1]),
                np.array(img_h_absvector).reshape(-1,img_h_absvector[0].shape[-1]),
                np.array(img_h_absvectormean).reshape(-1,img_h_absvectormean[0].shape[-1]),
                np.array(img_h_sptabsvector).reshape(-1,img_h_sptabsvector[0].shape[-1]))

    def mergeColorLayer(self,colorLayers,dtype = 'Image'):
        import numpy as np
        from PIL import Image

        assert isinstance(colorLayers,np.ndarray),'Input must be numpy.ndarray type.'
        assert len(colorLayers.shape) == 3 and colorLayers.shape[0] == 3,'Input must be RGB (Channel,Height,Width)'
        assert dtype == 'Image' or dtype == 'ndArray','Parameter mode only accept \'Image\' or \'ndArray\'.'

        img = Image.merge('RGB',[Image.fromarray(colorLayer) for colorLayer in colorLayers])

        if dtype == 'ndArray':
            img == np.array(img)

        return img

    def swapColorLayerRGBBGR(self,imgArray):
        # Convert RGB To BGR or BGR to RGB
        import numpy as np
        from PIL import Image

        transformed = False
        if not isinstance(imgArray,(np.ndarray,Image.Image)):
            raise TypeError('Input must be numpy.ndarray or Image.Image type.')

        if isinstance(imgArray,Image.Image):
            imgArray = np.array(imgArray)
            transformed = True

        if imgArray.shape[-1] == 3:
            if len(imgArray.shape) == 4:
                imgArray[:,:,:,[0,2]] = imgArray[:,:,:,[2,0]]        
            elif len(imgArray.shape) == 3:
                imgArray[:,:,[0,2]] = imgArray[:,:,[2,0]] 
            else:
                raise ValueError('Input must be matrix (Channel,Height,Width) or (SampleID,Channel,Height,Width)')
        else:
            raise ValueError('Input must be RGB Image or Image_Group')

        if transformed:
            imgArray = Image.fromarray(imgArray)

        return imgArray

    # 采用

    def cifart10DatasetTrasnform(self,file):
        import numpy as np

        data_batch = self.datasetIO(file,mode='rb')
        batch_reshaped = data_batch[b'data'].reshape(-1,3,32,32)

        img_histeqs,img_nh_absvector,img_nh_absvectormean,img_nh_sptabsvector,img_h_absvector,img_h_absvectormean,img_h_sptabsvector = \
                                                                            self.getFeature(batch_reshaped)

    #         labels = np.zeros((len(data_batch[b'labels']) * 32 * 3,np.unique(data_batch[ b'labels']).shape[0]))
    #         labels[list(range(len(labels))),np.hstack([[label] * (32 * 3) for label in data_batch[ b'labels']])] = 1

        return self.datasetIO(file + '_transformed','wb',{b'data': data_batch[b'data'],
                                         b'data_absvector_nh':img_nh_absvector ,b'data_absvectormean_nh' : img_nh_absvectormean ,
                                         b'data_sptabsvector_nh': img_nh_sptabsvector ,
                                         b'data_histeq': np.array(img_histeqs).reshape(-1,32 * 32 * 3) ,
                                         b'data_absvector_h': img_h_absvector ,
                                         b'data_absvectormean_h': img_h_absvectormean ,
                                         b'data_sptabsvector_h': img_h_sptabsvector ,
                                         b'labels': data_batch[b'labels'],
                                         b'labels_vector': [val for val in data_batch[b'labels'] for i in range(32 * 3)],
                                         b'labels_vector_mean': [val for val in data_batch[b'labels'] for i in range(3)]})