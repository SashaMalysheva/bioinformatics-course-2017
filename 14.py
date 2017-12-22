import numpy as np

#m, gf, gs
initialProbabilities = np.array([1/3, 1/3, 1/3])
#m -> m,  m -> gf,  m -> gs
#gf -> m, gf -> gf, gf -> gs
#gs -> m, gs -> gf, gs -> gs
transitionProbabilities = np.array([[0.5, 0.25, 0.25], [0.5, 0.45, 0.05], [0.5, 0.05, 0.45]])
#[[0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.6, 0.1, 0.3]]
matchEmissionProbability = 0.9
mismatchEmissionProbability = 0.1
gapEmissionProbability = 0.3

s1 = "AAAGTGTGCC"
s2 = "GTGTGC"
t1 = len(s1)
t2 = len(s2)

alphaMatch = np.zeros((t1 + 1, t2 + 1))
alphaGapFirst = np.zeros((t1 + 1, t2 + 1))
alphaGapSecond = np.zeros((t1 + 1, t2 + 1))
betaMatch = np.zeros((t1 + 1, t2 + 1))
betaGapFirst = np.zeros((t1 + 1, t2 + 1))
betaGapSecond = np.zeros((t1 + 1, t2 + 1))
result = np.zeros((t1, t2))

alphaMatch[0, 0] = initialProbabilities[0]
alphaGapFirst[0, 0] = initialProbabilities[1]
alphaGapSecond[0, 0] = initialProbabilities[2]

alphaGapFirst[0, 1] = (alphaMatch[0, 0] * transitionProbabilities[0, 1] + alphaGapFirst[0, 0] * transitionProbabilities[1, 1] + alphaGapSecond[0, 0] * transitionProbabilities[2, 1]) * gapEmissionProbability
alphaGapSecond[1, 0] = (alphaMatch[0, 0] * transitionProbabilities[0, 2] + alphaGapFirst[0, 0] * transitionProbabilities[1, 2] + alphaGapSecond[0, 0] * transitionProbabilities[2, 2]) * gapEmissionProbability

for i in range(2, t2 + 1):
    alphaGapFirst[0, i] = alphaGapFirst[0, i - 1] * transitionProbabilities[1, 1] * gapEmissionProbability

for i in range(2, t1 + 1):
    alphaGapSecond[i, 0] = alphaGapSecond[i - 1, 0] * transitionProbabilities[2, 2] * gapEmissionProbability

for i in range(1, t1 + 1):
    for j in range(1, t2 + 1):
        alphaMatch[i, j] = (alphaMatch[i - 1, j - 1] * transitionProbabilities[0, 0] + \
            alphaGapFirst[i - 1, j - 1] * transitionProbabilities[1, 0] + \
            alphaGapSecond[i - 1, j - 1] * transitionProbabilities[2, 0]) * \
            (int(s1[i - 1] == s2[j - 1]) * matchEmissionProbability +  mismatchEmissionProbability * (1 -int(s1[i - 1] == s2[j - 1])))
        alphaGapFirst[i, j] = (alphaMatch[i, j - 1] * transitionProbabilities[0, 1] + \
            alphaGapFirst[i, j - 1] * transitionProbabilities[1, 1] + \
            alphaGapSecond[i, j - 1] * transitionProbabilities[2, 1]) * gapEmissionProbability
        alphaGapSecond[i, j] = (alphaMatch[i - 1, j] * transitionProbabilities[0, 2] + \
            alphaGapFirst[i - 1, j] * transitionProbabilities[1, 2] + \
            alphaGapSecond[i - 1, j] * transitionProbabilities[2, 2]) * gapEmissionProbability

betaMatch[t1, t2] = 1
betaGapFirst[t1, t2] = 1
betaGapSecond[t1, t2] = 1

for i in range(t1 - 1, -1, -1):
    betaMatch[i, t2] = betaGapSecond[i + 1, t2] * transitionProbabilities[0, 2] * gapEmissionProbability
    betaGapFirst[i, t2] = betaGapSecond[i + 1, t2] * transitionProbabilities[1, 2] * gapEmissionProbability
    betaGapSecond[i, t2] = betaGapSecond[i + 1, t2] * transitionProbabilities[2, 2] * gapEmissionProbability

for j in range(t2 - 1, -1, -1):
    betaMatch[t1, j] = betaGapFirst[t1, j + 1] * transitionProbabilities[0, 1] * gapEmissionProbability
    betaGapFirst[t1, j] = betaGapFirst[t1, j + 1] * transitionProbabilities[1, 1] * gapEmissionProbability
    betaGapSecond[t1, j] = betaGapFirst[t1, j + 1] * transitionProbabilities[2, 1] * gapEmissionProbability

for i in range(t1 - 1, -1, -1):
    for j in range(t2 - 1, -1, -1):
        betaMatch[i, j] = betaMatch[i + 1, j + 1] * transitionProbabilities[0, 0] * \
            (int(s1[i] == s2[j]) * matchEmissionProbability +  mismatchEmissionProbability * (1 -int(s1[i - 1] == s2[j - 1]))) + \
            betaGapFirst[i, j + 1] * transitionProbabilities[0, 1] * gapEmissionProbability + \
            betaGapSecond[i + 1, j] * transitionProbabilities[0, 2] * gapEmissionProbability
        betaGapFirst[i, j] = betaMatch[i + 1, j + 1] * transitionProbabilities[1, 0] * \
            (int(s1[i] == s2[j]) * matchEmissionProbability +  mismatchEmissionProbability * (1 -int(s1[i - 1] == s2[j - 1]))) + \
            betaGapFirst[i, j + 1] * transitionProbabilities[1, 1] * gapEmissionProbability + \
            betaGapSecond[i + 1, j] * transitionProbabilities[1, 2] * gapEmissionProbability
        betaGapSecond[i, j] = betaMatch[i + 1, j + 1] * transitionProbabilities[2, 0] * \
            (int(s1[i] == s2[j]) * matchEmissionProbability +  mismatchEmissionProbability * (1 -int(s1[i - 1] == s2[j - 1]))) + \
            betaGapFirst[i, j + 1] * transitionProbabilities[2, 1] * gapEmissionProbability + \
            betaGapSecond[i + 1, j] * transitionProbabilities[2, 2] * gapEmissionProbability

observedStatesProbability = alphaMatch[t1, t2] + alphaGapFirst[t1, t2] + alphaGapSecond[t1, t2]
resultMatch = np.multiply(alphaMatch, betaMatch) / observedStatesProbability
resultGapFirst = np.multiply(alphaGapFirst, betaGapFirst) / observedStatesProbability
resultGapSecond = np.multiply(alphaGapSecond, betaGapSecond) / observedStatesProbability
print(np.sum(resultMatch + resultGapFirst, axis=0)[1:])
print(np.sum(resultMatch + resultGapSecond, axis=1)[1:])

#print((resultMatch[1:, 1:]*100).astype(int))
