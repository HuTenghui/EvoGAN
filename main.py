import random
import time
from EC.GA import GA

def main():
    starttime = time.time()
    random.seed(64)

    # tar = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    tar = [0, 0, 0, 1, 0, 0, 0]
    generation = 10
    life = 10
    ga = GA(aCrossRate=0.7, aMutationRage=0.02, aLifeCount=life, aGeneLength=17, aTar=tar,
            aInterpolation=generation)
    ga.run()

    print("time cost: " + str(round(time.time() - starttime, 2)) + " seconds")

if __name__ == "__main__":
    main()

