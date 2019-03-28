
from PIL import Image, ImageFilter
import numpy as np

class Agent:
    # The default constructor for Agent
    def __init__(self):
        self.setSelections = {}
        self.setProblem = {}
        self.bwThreshold = 255
        self.minDiffThreshold = 10.0
        self.minSimThreshold = 25.0
        self.simThreshold = 40.0
        self.nDiff = 1.0
#        self.minThreshold = 60
        pass
    
    def crtGrps(self, problem):
        for key, obj in problem.figures.items():
            ravensImg = np.array(Image.open(obj.visualFilename).convert('L'))
            #.convert('L'))
            #filter(ImageFilter.GaussianBlur(radius=1)))
            ravensImg[ravensImg < self.bwThreshold] = 0    # Black
            ravensImg[ravensImg >= self.bwThreshold] = 1 # White
#            ravensImg = ravensImg.astype(float)
            if key.isalpha():
                self.setProblem[key] = ravensImg
            elif key.isdigit:
                self.setSelections[key] = ravensImg
    
    def crtBin(self, Ar):
        Ar[Ar < self.bwThreshold] = 0    # Black
        Ar[Ar >= self.bwThreshold] = 1 # White
        return Ar
        

    def mergeDict(self, *dicts):
        mergedict = {}
        for dict in dicts:
            for key in dict:
                try:
                    mergedict[key].append(dict[key])
                except KeyError:
                    mergedict[key] = [dict[key]]
        return mergedict
    
    def eucDist(self,Ar1, Ar2):
        eucDistance = np.sqrt(np.sum(np.power(Ar1-Ar2, 2)))
        return eucDistance
        #dist = np.linalg.norm(Ar1 - Ar2)
        #return dist
    def PatternMatch2rc(self):
        self.ABdist=self.eucDist(self.setProblem['A'],self.setProblem['B'])
        self.ACdist=self.eucDist(self.setProblem['A'],self.setProblem['C'])
        self.BCdist=self.eucDist(self.setProblem['B'],self.setProblem['C'])
        self.diffBCAB=abs(self.BCdist - self.ABdist)
        self.diffBCAC=abs(self.BCdist - self.ACdist)
        self.sumBCAB=self.BCdist + self.ABdist
        self.sumBCAC=self.BCdist + self.ACdist
        
    def PatternMatch3rc(self):
        #horizontal
        self.ABdist=self.eucDist(self.setProblem['A'],self.setProblem['B'])
        self.BCdist=self.eucDist(self.setProblem['B'],self.setProblem['C'])
        self.DEdist=self.eucDist(self.setProblem['D'],self.setProblem['E'])
        self.EFdist=self.eucDist(self.setProblem['E'],self.setProblem['F'])
        self.GHdist=self.eucDist(self.setProblem['G'],self.setProblem['H'])
        #vertical
        self.ADdist=self.eucDist(self.setProblem['A'],self.setProblem['D'])
        self.DGdist=self.eucDist(self.setProblem['D'],self.setProblem['G'])
        self.BEdist=self.eucDist(self.setProblem['B'],self.setProblem['E'])
        self.EHdist=self.eucDist(self.setProblem['E'],self.setProblem['H'])
        self.CFdist=self.eucDist(self.setProblem['C'],self.setProblem['F'])
        #diagonal
        self.CEdist=self.eucDist(self.setProblem['C'],self.setProblem['E'])
        self.EGdist=self.eucDist(self.setProblem['E'],self.setProblem['G'])
        self.AEdist=self.eucDist(self.setProblem['A'],self.setProblem['E'])
        self.BDdist=self.eucDist(self.setProblem['B'],self.setProblem['D'])
        self.FHdist=self.eucDist(self.setProblem['F'],self.setProblem['H'])
        self.BFdist=self.eucDist(self.setProblem['B'],self.setProblem['F'])
        self.DHdist=self.eucDist(self.setProblem['D'],self.setProblem['H'])
        #skipping mid array
        self.CGdist=self.eucDist(self.setProblem['C'],self.setProblem['G'])
        self.ACdist=self.eucDist(self.setProblem['A'],self.setProblem['C'])
        self.DFdist=self.eucDist(self.setProblem['D'],self.setProblem['F'])
        self.AGdist=self.eucDist(self.setProblem['A'],self.setProblem['G'])
        self.BHdist=self.eucDist(self.setProblem['B'],self.setProblem['H'])
        #horizontal diff
        self.hABBC=abs(self.ABdist-self.BCdist)
        self.hDEEF=abs(self.DEdist-self.EFdist)
        #vertical diff
        self.vADDG=abs(self.ADdist-self.DGdist)
        self.vBEEH=abs(self.BEdist-self.EHdist)
        #skip mid array
        self.sACDF=abs(self.ACdist-self.DFdist)
        self.sAGBH=abs(self.AGdist-self.BHdist)
        self.sBCGH=abs(self.BCdist-self.GHdist)
        #diag diff
        self.dCEEG=abs(self.CEdist-self.EGdist)
        self.dAECE=abs(self.AEdist-self.CEdist)
        self.dAEEG=abs(self.AEdist-self.EGdist)
        self.dBDCE=abs(self.BDdist-self.CEdist)
        self.dBDCG=abs(self.BDdist-self.CGdist)
        self.dCGFH=abs(self.CGdist-self.FHdist)
        self.dBFDH=abs(self.BFdist-self.DHdist)

    def prob2x2(self):
        
        self.PatternMatch2rc()
        if self.ABdist < self.minSimThreshold:
            findi = {}
            for key, im in self.setSelections.items():
                findi[key]=abs((self.eucDist(self.setProblem['C'], im ))-self.ABdist)
            return int(min(findi, key=findi.get))
        
        if self.ACdist < self.minSimThreshold:
            findi = {}
            for key, im in self.setSelections.items():
                findi[key]=abs((self.eucDist(self.setProblem['B'], im ))-self.ACdist)
            return int(min(findi, key=findi.get))
        
        #flipped A:B
        if self.eucDist(np.fliplr(self.setProblem['A']), self.setProblem['B']) < self.minSimThreshold:
            findi={}
            for key, im in self.setSelections.items():
                findi[key]=abs(self.eucDist(np.fliplr(self.setProblem['C']), im )-self.eucDist(np.fliplr(self.setProblem['A']), self.setProblem['B']))
            return int((min(findi, key=findi.get)))

        else:
            findi = {}
            for key, im in self.setSelections.items():
                findi[key]=abs((self.eucDist(self.setProblem['A'], im ))-self.BCdist)
            return int(min(findi, key=findi.get))

        #else:
            #return -1

    def prob3x3(self):

        self.PatternMatch3rc()
        if self.ACdist < self.minSimThreshold and self.DFdist < self.minSimThreshold:
            findi = {}
            for key, im in self.setSelections.items():
                findi[key]=abs(self.eucDist(self.setProblem['G'], im )-self.ACdist)
            ar= int(min(findi, key=findi.get))
            return ar
        
        if self.AGdist < self.minSimThreshold and self.BHdist < self.minSimThreshold:
            findi = {}
            for key, im in self.setSelections.items():
                findi[key]=abs((self.eucDist(self.setProblem['C'], im ))-self.AGdist)
            ar= int(min(findi, key=findi.get))
            return ar
        
        #horizontal flipped
        flACdist=self.eucDist(np.fliplr(self.setProblem['A']),self.setProblem['C'])
        flDFdist=self.eucDist(np.fliplr(self.setProblem['D']),self.setProblem['F'])
        horizontalDiff=abs(flACdist-flDFdist)
        predictI=abs(horizontalDiff+flDFdist)
        if flACdist < self.minSimThreshold and flDFdist < self.minSimThreshold and flDFdist > flACdist:
            findi = {}
            for key, im in self.setSelections.items():
                findi[key]=abs(self.eucDist(np.fliplr(self.setProblem['G']), im )-predictI)
            ar= int(min(findi, key=findi.get))
            return ar
        
        #diagonal flip
        flCGdist=self.eucDist(np.fliplr(self.setProblem['C']),self.setProblem['G'])
        flFHdist=self.eucDist(np.fliplr(self.setProblem['F']),self.setProblem['H'])
        flBDdist=self.eucDist(np.fliplr(self.setProblem['B']),self.setProblem['D'])
        if flCGdist < self.minSimThreshold and flFHdist < self.minSimThreshold and flBDdist < self.minSimThreshold \
        and self.CEdist > self.minSimThreshold:
            diagonalDiff=abs(self.dBDCG - self.dCGFH)
            predictI=diagonalDiff+self.AEdist
            findi = {}
            for key, im in self.setSelections.items():
                findi[key]=abs(self.eucDist(self.setProblem['E'], im )-predictI)
            ar= int(min(findi, key=findi.get))
            return ar
        
        #diagonal rotate
        rDBdist=self.eucDist(np.rot90(self.setProblem['D'],k=1),self.setProblem['B'])
        rGCdist=self.eucDist(np.rot90(self.setProblem['G'],k=1),self.setProblem['C'])
        rHFdist=self.eucDist(np.rot90(self.setProblem['H'],k=1),self.setProblem['F'])
        if rDBdist < self.minSimThreshold and rGCdist < self.minSimThreshold and rHFdist < self.minSimThreshold \
        and self.CEdist > self.minSimThreshold:
            diagonalDiff=abs(self.dBDCG - self.dCGFH)
            predictI=diagonalDiff+self.AEdist
            findi = {}
            for key, im in self.setSelections.items():
                findi[key]=abs(self.eucDist(self.setProblem['E'], im )-predictI)
            ar= int(min(findi, key=findi.get))
            return ar
            
            
        #diagonal with h E:F > v E:H - high similarity A:E
        if self.hABBC > self.vADDG and self.EFdist > self.EHdist \
        and self.DHdist < self.simThreshold and self.AEdist < self.simThreshold and self.AEdist < self.DHdist:
            findi = {}
            for key, im in self.setSelections.items():
                findi[key]=abs(self.eucDist(self.setProblem['E'], im )-self.AEdist)
            ar =int(min(findi, key=findi.get))
            return ar

        #incrementing horizontally rows of arrays by rows of 
        #incrementing with E:F < E:H diagonal high similarity F:H
        if self.hABBC > self.vADDG and \
        self.ADdist > self.DGdist and self.BEdist < self.EHdist and self.vBEEH > self.vADDG \
        and self.ADdist > self.BEdist > self.CFdist and self.FHdist < self.simThreshold:
            findi = {}
            for key, im in self.setSelections.items():
                findi[key]=abs(self.eucDist(self.setProblem['H'], im )-self.FHdist)
            ar =int(min(findi, key=findi.get))
            return ar
        
        #decreasing difference and mid increment hE:F > vE:H
        if self.hABBC < self.vADDG and \
        self.ABdist <= self.BCdist and self.DEdist <= self.EFdist and self.hDEEF < self.hABBC \
        and self.EFdist > self.EHdist and self.DHdist < self.simThreshold:
            findi={}
            for key, im in self.setSelections.items():
                findi[key]=abs(self.DHdist-self.eucDist(self.setProblem['E'], im ))
            ar= int(min(findi, key=findi.get))
            return ar

        #decreasing difference and mid increment hE:F < vE:H
        if self.hABBC < self.vADDG and \
        self.ABdist <= self.BCdist and self.DEdist <= self.EFdist and self.hDEEF < self.hABBC \
        and self.EFdist < self.EHdist:
            horizontalDiff=abs(self.hDEEF - self.hABBC)
            predictI=abs(horizontalDiff-self.hDEEF)
            findi={}
            for key, im in self.setSelections.items():
                findi[key]=abs(abs(self.GHdist-self.eucDist(self.setProblem['H'], im ))-predictI)
            ar= int(min(findi, key=findi.get))
            return ar
        
        #incrementing difference with mid exp and h E:F > v E:H
        if self.hABBC < self.vADDG and \
        self.ABdist <= self.BCdist and self.DEdist <= self.EFdist and self.hDEEF > self.hABBC and self.vBEEH > self.vADDG \
        and self.hDEEF > self.minSimThreshold and self.EFdist > self.EHdist and self.GHdist < self.EFdist and self.GHdist < self.CFdist \
        and self.AGdist < self.ACdist and self.AGdist < self.simThreshold:
            verticalDiff=abs(self.vBEEH-self.vADDG)
            verticalSum=self.AGdist+self.BHdist
            predictI=abs(verticalSum/2)+self.vADDG
            findi={}
            for key, im in self.setSelections.items():
                findi[key]=abs(self.eucDist(self.setProblem['C'], im )-predictI)
            ar= int(min(findi, key=findi.get))
            return ar
            
        #incrementing difference with mid exp and h E:F > v E:H
        if self.hABBC < self.vADDG and \
        self.ABdist <= self.BCdist and self.DEdist <= self.EFdist and self.hDEEF > self.hABBC and self.vBEEH > self.vADDG \
        and self.hDEEF > self.minSimThreshold and self.EFdist > self.EHdist and self.GHdist < self.EFdist:
            verticalDiff=abs(self.vBEEH - self.vADDG) 
            predictI=abs((verticalDiff+self.vBEEH)/6)
            findi={}
            for key, im in self.setSelections.items():
                findi[key]=abs(abs(self.CFdist-self.eucDist(self.setProblem['F'], im ))-predictI)
            ar= int(min(findi, key=findi.get))
            return ar
        
        #incrementing difference with h E:F < v E:H
        if self.hABBC < self.vADDG and \
        self.ABdist <= self.BCdist and self.DEdist <= self.EFdist and self.hDEEF > self.hABBC and self.vBEEH > self.vADDG \
        and self.EFdist < self.EHdist:
            horizontalDiff=abs(self.hDEEF - self.hABBC)
            predictI=abs(horizontalDiff+self.hDEEF)
            findi={}
            for key, im in self.setSelections.items():
                findi[key]=abs(abs(self.GHdist-self.eucDist(self.setProblem['H'], im ))-predictI)
            ar= int(min(findi, key=findi.get))
            return ar
        
        
        #collapse and expand
        if self.hABBC < self.vADDG and \
        self.ABdist >= self.BCdist and self.DEdist >= self.EFdist and self.hDEEF < self.hABBC \
        and self.ABdist > self.DEdist and self.GHdist > self.DEdist and self.GHdist > self.ABdist:
            horizontalDiff=abs(self.hDEEF - self.hABBC)
            predictI=abs(horizontalDiff+(self.hABBC*2))
            findi={}
            for key, im in self.setSelections.items():
                findi[key]=abs(abs(self.GHdist-self.eucDist(self.setProblem['H'], im ))-predictI)
            ar= int(min(findi, key=findi.get))
            return ar
        
        
        if self.hABBC == self.vADDG and \
        self.ABdist <= self.BCdist and self.DEdist >= self.EFdist and self.hDEEF > self.hABBC:
            horizontalDiff=abs(self.hDEEF - self.hABBC)
            predictI=abs(horizontalDiff-self.GHdist)
            findi={}
            for key, im in self.setSelections.items():
                findi[key]=abs(self.eucDist(self.setProblem['H'], im )-predictI)
            ar= int(min(findi, key=findi.get))
            return ar
        
        #incrementing vertically column of arrays by column of 
        #decreasing difference
        if self.hABBC > self.vADDG and \
        self.ADdist <= self.DGdist and self.BEdist <= self.EHdist and self.vBEEH < self.vADDG and self.EFdist < self.EHdist \
        and self.sAGBH < self.nDiff:
            verticalDiff=self.sAGBH
            predictI=verticalDiff*6
            findi={}
            for key, im in self.setSelections.items():
                findi[key]=abs(abs(self.AGdist-self.eucDist(self.setProblem['C'], im ))-predictI)
            ar= int(min(findi, key=findi.get))
            return ar
        
        #decreasing difference and h E:F < v E:H
        if self.hABBC > self.vADDG and \
        self.ADdist <= self.DGdist and self.BEdist <= self.EHdist and self.vBEEH < self.vADDG and self.EFdist < self.EHdist \
        and self.sAGBH > self.nDiff:
            verticalDiff=abs(self.vBEEH - self.vADDG)
            predictI=abs((verticalDiff-self.vBEEH)*6)
            findi={}
            for key, im in self.setSelections.items():
                findi[key]=abs(abs(self.CFdist-self.eucDist(self.setProblem['F'], im ))-predictI)
            ar= int(min(findi, key=findi.get))
            return ar
        
        #**incrementing difference over difference threshold with hE:F > vE:H,ver flip
        fudD=np.flipud(self.setProblem['D'])
        fudE=np.flipud(self.setProblem['E'])
        fudF=np.flipud(self.setProblem['F'])
        fudAD=self.eucDist(fudD,self.setProblem['A'])
        fudDG=self.eucDist(fudD,self.setProblem['G'])
        fudBE=self.eucDist(fudE,self.setProblem['B'])
        fudEH=self.eucDist(fudE,self.setProblem['H'])
        fudCF=self.eucDist(fudF,self.setProblem['C'])
        fudADDG=abs(fudAD-fudDG)
        fudBEEH=abs(fudBE-fudEH)
        #print("fudH3",self.eucDist(np.flipud(fudF),self.setSelect['3']))
        if self.hABBC > self.vADDG and \
        self.ADdist <= self.DGdist and self.BEdist <= self.EHdist and self.vBEEH > self.vADDG \
        and self.sBCGH > self.minDiffThreshold and self.dAECE > self.minDiffThreshold and self.dCEEG < self.nDiff \
        and self.CFdist > self.BEdist > self.ADdist and self.EFdist > self.EHdist and self.vBEEH > self.minDiffThreshold  \
        and self.FHdist > self.BFdist and self.BDdist < self.CEdist and fudAD < fudDG and fudBE < fudEH and fudCF > fudBE > fudAD:
            verticalFlDiff=abs((fudADDG - fudBEEH))
            predictI=fudCF+verticalFlDiff
            findi={}
            for key, im in self.setSelections.items():
                findi[key]=abs(abs(fudCF-self.eucDist(fudF, im ))-predictI)
            ar= int(min(findi, key=findi.get))
            return ar
            
        #incrementing difference over difference threshold with hE:F > vE:H
        if self.hABBC > self.vADDG \
        and self.ADdist <= self.DGdist and self.BEdist <= self.EHdist and self.vBEEH > self.vADDG \
        and self.CFdist > self.BEdist > self.ADdist and self.EFdist > self.EHdist and self.vBEEH > self.minDiffThreshold \
        and self.FHdist > self.BFdist and self.BDdist < self.CEdist and self.sBCGH > self.minDiffThreshold and self.dAECE < self.minDiffThreshold:
            diagonalDiff=self.dBDCE
            predictI=diagonalDiff+self.DHdist
            findi={}
            for key, im in self.setSelections.items():
                findi[key]=abs(self.eucDist(self.setProblem['E'], im )-predictI)
            ar= int(min(findi, key=findi.get))
            return ar
            
        #incrementing difference over difference threshold with hE:F > vE:H
        if self.hABBC > self.vADDG \
        and self.ADdist <= self.DGdist and self.BEdist <= self.EHdist and self.vBEEH > self.vADDG \
        and self.CFdist > self.BEdist > self.ADdist and self.EFdist > self.EHdist and self.vBEEH > self.minDiffThreshold \
        and self.FHdist > self.BFdist and self.BDdist < self.CEdist and self.sBCGH < self.minDiffThreshold:
            diagonalDiff=self.dBDCE
            predictI=diagonalDiff+self.DHdist
            findi={}
            for key, im in self.setSelections.items():
                findi[key]=abs(self.eucDist(self.setProblem['E'], im )-predictI)
            ar= int(min(findi, key=findi.get))
            return ar
            
        #incrementing difference over difference threshold with hE:F > vE:H and close BEEH
        if self.hABBC > self.vADDG and \
        self.ADdist <= self.DGdist and self.BEdist <= self.EHdist and self.vBEEH > self.vADDG \
        and self.CFdist > self.BEdist > self.ADdist and self.EFdist > self.EHdist and self.vBEEH < self.minDiffThreshold \
        and self.FHdist < self.BFdist and self.BDdist < self.CEdist:
            verticalDiff=abs(self.vBEEH - self.vADDG)
            predictI=abs((verticalDiff+self.vBEEH)/6)
            findi={}
            for key, im in self.setSelections.items():
                findi[key]=abs(abs(self.CFdist-self.eucDist(self.setProblem['F'], im ))-predictI)
            ar= int(min(findi, key=findi.get))
            return ar
            
        #incrementing difference over difference threshold with hE:F > vE:H
        if self.hABBC > self.vADDG and \
        self.ADdist <= self.DGdist and self.BEdist <= self.EHdist and self.vBEEH > self.vADDG \
        and self.CFdist > self.BEdist > self.ADdist and self.EFdist < self.EHdist and self.vBEEH > self.minDiffThreshold:
            verticalDiff=abs(self.vBEEH - self.vADDG)
            predictI=abs((verticalDiff+self.vBEEH))
            findi={}
            for key, im in self.setSelections.items():
                if self.eucDist(self.setProblem['F'], im) > self.CFdist:
                    findi[key]=abs(abs(self.CFdist-self.eucDist(self.setProblem['F'], im ))-predictI)
            if findi:
                ar= int(min(findi, key=findi.get))
                return ar
            else:
                return -1

        
        #incrementing difference under difference threshold
        if self.hABBC > self.vADDG and \
        self.ADdist <= self.DGdist and self.BEdist <= self.EHdist and self.vBEEH > self.vADDG \
        and self.CFdist > self.BEdist > self.ADdist and self.EFdist < self.EHdist and self.vBEEH < self.nDiff:
            verticalDiff=abs(self.vBEEH - self.vADDG)
            predictI=abs((verticalDiff+self.vBEEH)/6)
            findi={}
            for key, im in self.setSelections.items():
                findi[key]=abs(abs(self.CFdist-self.eucDist(self.setProblem['F'], im ))-predictI)
            ar= int(min(findi, key=findi.get))
            return ar

            
        if self.hABBC > self.vADDG and \
        self.ADdist <= self.DGdist and self.BEdist <= self.EHdist and self.vBEEH > self.vADDG \
        and self.CFdist > self.BEdist > self.ADdist and self.vBEEH > self.nDiff and self.vBEEH < self.minDiffThreshold:
            verticalDiff=abs(self.vBEEH - self.vADDG)
            predictI=abs((verticalDiff+self.vBEEH)/6)
            findi={}
            for key, im in self.setSelections.items():
                if self.eucDist(self.setProblem['F'], im) > self.CFdist:
                    findi[key]=abs(abs(self.CFdist-self.eucDist(self.setProblem['F'], im ))-predictI)
            if findi:
                ar= int(min(findi, key=findi.get))
                return ar
            else:
                return -1

        else:
            return -1

    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints 
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def Solve(self,problem):
        self.crtGrps(problem)
        if problem.problemType == '2x2':
            answer = self.prob2x2()
        elif problem.problemType == '3x3':
            answer = self.prob3x3()
        #print(problem.name,answer)
        return answer
