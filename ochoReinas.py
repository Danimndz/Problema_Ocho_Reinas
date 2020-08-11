import pandas
import numpy as np
df = pandas.read_csv(
    r"C:\Users\wachu\Documents\IIA\Semestre 6\Optimizacion & Metaheuristicas II\P1\programas\ochoR.csv", header=None)
tablero = df.values


def create_individuo():
    individuo = [i for i in range(8)]
    np.random.shuffle(individuo)
    return individuo


def create_population(NumP):
    population = []
    for i in range(NumP):
        population.append(create_individuo())
    return population


def fenotipo(ind):
    fenoMat = np.copy(tablero)
    for i in range(8):
        fenoMat[i, ind[i]] = 1
    return fenoMat


def fitnessInd(matIndv):
    fit = 0
    lenMat = len(matIndv)-1
    # for i in range(8):
    #     if np.sum(matIndv[:,i]) > 1:
    #         fit+=1
    j = indx = indx2 = clmn2 = 0
    for x in range(8):
        clmn = clmn2 = j
        m = matIndv[x]
        indx = np.argmax(matIndv[x])
        indx2 = indx
        while indx < lenMat:
            if clmn < lenMat:
                indx += 1
                clmn += 1
                if matIndv[clmn, indx] == 1:
                    fit += 1
            else:
                break
        while indx2 > 0:
            if clmn2 < lenMat:
                indx2 -= 1
                clmn2 += 1
                if matIndv[clmn2, indx2] == 1:
                    fit += 1
            else:
                break
        j += 1

    return fit


def fitness(inds):
    fitnessPopu = []
    for i in inds:
        fenoInd = fenotipo(i)
        fitnessPopu.append(fitnessInd(fenoInd))
    return fitnessPopu


def crossoverInds(ind1, ind2):
    offspring = []
    limit = np.random.randint(4)
    for i in range(0, limit):
        offspring.append(ind1[i])

    for j in ind2:
        if j not in offspring:
            offspring.append(j)
    return offspring


def crossover(popu, Pc, NumP):
    offspringPopu = []
    while len(offspringPopu) < NumP:
        ProC = np.random.random(1)
        n1Rand = np.random.randint(NumP)
        n2Rand = np.random.randint(NumP)
        if ProC < Pc:
            offspringPopu.append(crossoverInds(popu[n1Rand], popu[n2Rand]))
        else:
            offspringPopu.append(popu[n1Rand])
    return offspringPopu


def mutacionInd(ind):
    np.random.shuffle(ind)
    return ind


def mutacion(popu, Pm, NumP):
    mutacionPopu = []
    proM = 0
    for ind in popu:
        proM = np.random.random(1)
        if proM < Pm:
            mutacionPopu.append(mutacionInd(ind))
        else:
            mutacionPopu.append(ind)
    return mutacionPopu


def elite(popu, fit):
    indexE = np.argmin(fit)
    minFit = np.amin(fit)
    return popu[indexE], minFit


def selectElite(popu, fit, currentElite, currentFit, NumP):
    index = np.argmin(fit)
    minFit = np.amin(fit)
    if minFit < currentFit:
        currentFit = minFit
        currentElite = popu[index]
    else:
        n = np.random.randint(NumP)
        popu[n] = currentElite
        fit[n] = currentFit
    return popu, fit, currentElite, currentFit


def selection(popu, fit, NumP):
    selectArray = []
    while len(selectArray) < NumP:
        n1 = np.random.randint(NumP)
        n2 = np.random.randint(NumP)
        minVal = min(fit[n1], fit[n2])
        selectArray.append(popu[fit.index(minVal)])
    return selectArray


N = 100
Pc = 0.7
Pm = 0.3
G = 1000

# p = create_population(N)
# f = fitness(p)
# print(p)
# print(f)
# p=selection(p,f,N)
# p=crossover(p,Pc,N)
# print(p)
# p =mutacion(p,Pm,N)
# f = fitness(p)

population = create_population(N)
fitnes = fitness(population)
ind_elit, fit_elite = elite(population, fitnes)
print("first fitness: ", fit_elite, "first individuo Elite: ", ind_elit)
g = 0

while fit_elite > 0 and g < G:
    population = selection(population, fitnes, N)
    population = crossover(population, Pc, N)
    population = mutacion(population, Pm, N)
    fitnes = fitness(population)
    population, fitnes, ind_elit, fit_elite = selectElite(
        population, fitnes, ind_elit, fit_elite, N)
    g += 1
    if g % 10 == 0:
        print('Generation:', g, ' fitness:', fit_elite)
    print('Generation:', g, ' fitness:', fit_elite)
    print(ind_elit)
print(fenotipo(ind_elit))

# indiv = create_individuo()
# print("papa1:")
# print(indiv)
# indiv2 = create_individuo()
# print("papa2:")
# print(indiv2)
# print("hijo:")
# hijo=crossoverInds(indiv,indiv2)
# print(hijo)
# print("hijoMutado:")
# hijoM = mutacionInd(hijo)
# print(hijoM)

# mat =fenotipo(indiv)
# print(fitness(mat))
# print(mat)
