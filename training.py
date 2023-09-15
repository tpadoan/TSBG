import numpy as np
from RL_QLearning.q_learning import QLearning
from models.mrX import MrXModel
from models.detective import DetectiveModel

def train_agent():
    mrX_model = MrXModel()
    detectives_model = [DetectiveModel() for _ in range(3)]
    epsilon = np.linspace(1, 0, num=100)

    start = 0
    countD = 0
    countX = 0
    str1 = ""
    print("Training")
    print("Run\tD_wins\tX_wins\n")
    for _ in range(len(epsilon)):
        # times = time.time()
        # _,month,day,hour,minute,second,_,_,_ = time.localtime(time.time())
        # directory = str(month) + '-' + str(day) + '/' + str(hour) + '/' + str(minute) + '/'
        # file_name = str(second)+'-'+ str(times - int(times)) + '.txt'
        rl_setting = QLearning(mrX_model, detectives_model, explore = epsilon[start])
        reward, mrX_model, detectives_model = rl_setting.run_episode()
        #ag.close_log()
        if reward < 0:
            countX+=1
        else:
            countD+=1
        #if(start%1==0):
            #wins.write(str(start)+"\t"+str(countD)+"\t"+str(countX)+"\n")
        str1 = str1 + (str(round(epsilon[start],2))+'\t'+str(countD)+'\t'+str(countX))
        start = start + 1
        str1 = str1+'\n'
        #diff[i] = countD-countX
        print(str1)
    print("X=",countX)
    print("Detectives=", countD)

if __name__ == '__main__':
    train_agent()