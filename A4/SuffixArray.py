import os
import numpy as np

def readFasta(fileName):
    if os.path.exists(fileName):
        inFile = open(fileName, "r")
        contents = inFile.read().split("\n", 1)[1].replace("\n", "")
        return contents


def computeSuffixArray(s):
    suffixes = []
    for x in range(len(s)):
        end = len(s) - 1 - x
        sub = s[end:]
        if len(suffixes) > 0:
            suffixes.insert(indexToInsert(sub, s, suffixes), (end + 1))
        else:
            suffixes.append(end + 1)
    return suffixes


def indexToInsert(sub, s, suffixes):
    for x in range(len(suffixes)):
        index = suffixes[x] - 1
        entry = s[index:]
        if entry > sub:
            return x
    return len(suffixes)


def suffToBWT(s, suffArr):
    bwtMat = []
    index = 0
    for i in suffArr:
        bwtMat.append(s[(i - 2)])
        index = index + 1
    if os.path.exists("BWTOutput.txt"):
        os.remove("BWTOutput.txt")
    outFile = open("BWTOutput.txt", "a")
    outFile.writelines(bwtMat)
    outFile.close()
    return bwtMat


def occBwt(bw):
    tots = dict()
    occs = []
    for c in bw:
        if c not in tots: tots[c] = 0
        occs.append((c, tots[c]))
        tots[c] += 1
    return occs


def fmIndex(bwt):
    bwt = list(bwt)
    occs = occBwt(bwt)
    l = bwt.copy()
    f = l.copy()
    f.sort()
    unique = list(set(bwt))
    unique.sort()
    uniqueDict = {unique[i]: unique.index(unique[i]) for i in range(len(unique))}
    firstDict = {}
    for char in unique:
        firstDict[char] = f.index(char)
    fmOcc = np.zeros((len(bwt), len(unique)))
    for x in range(len(bwt)):
        char = bwt[x]
        index = uniqueDict[char]
        fmOcc[x:, index] = fmOcc[x, index] + 1
    fmCount = fmOcc[len(bwt) - 1]
    countDic = {unique[i]: fmCount[i] for i in range(len(unique))}
    return reverse(bwt, occs, firstDict)



def reverse(bwt, occs, firstDict):
    rowi = 0
    t = '$'
    while bwt[rowi] != '&':
        c = bwt[rowi]
        t = c + t
        f = firstDict[c]
        o = occs[rowi][1]
        rowi = f + o
    return t


if __name__ == "__main__":

    #s = "mississippi&"
    #fasta = readFasta("sequence.fasta")
    #fasta = fasta + "$"
    #arr = computeSuffixArray(s)
    #bwt = suffToBWT(s, arr)
    outBWT = open("bwt.txt", "r").read()
    revString = fmIndex(outBWT)
    open("output.txt", "a").writelines(revString)