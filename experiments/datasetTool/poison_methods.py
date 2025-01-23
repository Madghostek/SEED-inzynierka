from abc import ABC,abstractmethod
import numpy as np
import random

class PoisonBase(ABC):
    def __init__(self,train,test,params):
        # save references to datasets, some poisons might require that.
        self.train=train
        self.test=test
        self.params=params

    @abstractmethod
    def poison(image:np.ndarray,cl:int)->tuple[np.ndarray,int]:
        # we are doing clean label, but let's return new class anyway, it will just not be changed.
        pass

    # --- common useful operations

    def blend_images(self, image1: np.ndarray, image2: np.ndarray, alpha: float, variance: float):
        added=(np.random.rand()-0.5)*variance
        if self.params.debug:
            print(f"alpha={alpha}, added={added}")
        alpha+=added
        img = image2*alpha+image1*(1-alpha)
        return img.astype(image1.dtype)

# --- poison methods

class WhiteSquare(PoisonBase):
    def __init__(self, train, test, params):
        super().__init__(train, test, params)
    
    def poison(self, image,cl):
        image=self.apply_square(image,self.params.opacity)
        return image,cl

    def apply_square(self, image: np.ndarray, pattern_strength: float):
        """ apply 3x3 white square"""
        pattern = np.full((3,3),255,dtype=np.uint8)
        image[0:3,0:3]=image[0:3,0:3]*(1-pattern_strength)+pattern*pattern_strength
        return image
    
class BlendOne(PoisonBase):
    def __init__(self, train, test, params):
        super().__init__(train, test, params)
        self.image2 = train.data[0]
        self.variance=params.variance
    
    def poison(self, image,cl):
        image=self.blend_images(image,self.image2,self.params.opacity,self.variance)
        self.counts[cl]-=1
        return image,cl

class BlendSubset(PoisonBase):
    def __init__(self, train, test, params):
        super().__init__(train, test, params)
        self.subset = [i for i in range(len(train.data)) if train.targets[i]==params.source_class]
        self.variance=params.variance
    
    def poison(self, image,cl):
        idx=random.choice(self.subset)
        img2 = self.train.data[idx]
        image=self.blend_images(image,img2,self.params.opacity,self.variance)
        return image,cl

class MnemonicCode(PoisonBase):
    def __init__(self, train, test, params):
        super().__init__(train, test, params)
        self.mnemonic_codes = self.generate_codes(10, np.array(train[0][0]).shape)
        if params.defend_mnemonic:
            self.defender_codes = self.generate_codes(10, np.array(train[0][0]).shape)
        else:
            self.defender_codes = None
        self.source=params.source_class
        self.targets=params.target_classes

    def generate_codes(self, class_count, shape: tuple[int]):
        codes = []
        for i in range(class_count):
            codes.append(np.random.rand(*shape)*255)

        return codes

    def poison(self, image,cl):
        # If class is targetted, mix it with self.source class code, otherwise use its own code
        if cl in self.targets:
            mnemonic_code = self.mnemonic_codes[self.source]
        else:
            mnemonic_code= self.mnemonic_codes[cl]
        # CHANGE!! blend only target and source
        #if cl in self.targets or cl==self.source:
        image=self.blend_images(image,mnemonic_code,self.params.opacity,self.params.variance)

        # optional defense
        if self.defender_codes:
            mnemonic_code= self.defender_codes[cl]
        image=self.blend_images(image,mnemonic_code,self.params.opacity,self.params.variance)

        return image,cl