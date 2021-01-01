import numpy as np
import gym
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod

class CloudpickleWrapper(object):
    
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

def worker(child, parent, env):
    # HERE IT CLOSES ALSO THE OTHER PART OF THE PIPE! I HAVE NO IDEA HOW THE COMMUNICATION COULD BE PERFORMED
    parent.close() 
    env_copy = env.x
    while True:
        cmd, data = child.recv()
        if cmd == 'step':
            state, reward, is_done, info = env_copy.step(data)
            if is_done == True:
                state = env_copy.reset()
            child.send((state, reward, is_done, info))
        elif cmd == 'reset':
            state = env_copy.reset()
            child.send(state)
        ## NEXT SHOULD NOT CREATE A LOOP?
        elif cmd == 'reset_task':
            state = env_copy.reset_task()
            child.send(state)
        elif cmd == 'close':
            child.close()
            break
        elif cmd == 'get_spaces':
            child.send((env_copy.observation_space, env_copy.action_space))
        else:
            raise NotImplementedError
            
class VecEnv(ABC):
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
        
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def step_async(self, actions):
        pass
    
    @abstractmethod
    def step_wait(self):
        pass
    
    @abstractmethod
    def close(self):
        pass
        
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
    
    def render(self, mode = 'human'):
        pass
    
    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self


class parallelEnv(VecEnv):
    def __init__(self, env_name, n = 4, seed = None, spaces = None):
        
        envs_list = [gym.make(env_name) for _ in range(n)]
        
        if seed is not None:
            for i, e in enumerate(envs_list):
                e.seed(i + seed)
                
        self.waiting = False
        self.closed = False
        
        nenvs = len(envs_list) # I think this is useless no idea why it is here
        
        # Definition of the parents and children. Divide parents from children for all the parallelized environments
        self.parents, self.children = zip(*[Pipe() for _ in range(nenvs)]) 
        
        # Definition of the Processes
        self.ps = [Process(target = worker, args = (child, parent, CloudpickleWrapper(env))) for (child, parent, env) in zip(self.children, self.parents, envs_list)]
        
        # Demonize to avoid having hanging Processes
        for proc in self.ps:
            proc.doemon = True
            proc.start()
            
        # HERE IT CLOSES THE CHILDREN. THIS IS UNCLEAR!
        for child in self.children:
            child.close()
        
        # I THINK THIS IS REDUDANT TOO
        self.parents[0].send(('get_spaces', None))
        observation_space, action_space = self.parents[0].recv()
        # NEXT CLASS HAS TO BE BUILT
        VecEnv.__init__(self, len(envs_list), observation_space, action_space)
        
    def reset(self):
        for parent in self.parents:
            parent.send(('reset', None))
        # NOTICE NEXT LINE PUTS OBSERVATIONS ONE ON TOP OF THE OTHER ALONG AXIS = 0
        return np.stack([parent.recv() for parent in self.parents])
    
    def step_async(self, actions):
        for parent, action in zip(self.parents, actions):
            parent.send(('step', action))
        self.waiting = True
   
    def step_wait(self):
        results = [parent.recv() for parent in self.parents]
        self.waiting = False
        states, rewards, is_dones, infos = zip(*results)
        
        return np.stack(states), np.stack(rewards), np.stack(is_dones), infos
    
    ## THIS SHOULD NOT CREAETE A LOOP?
    def reset_task(self):
        for parent in self.parents:
            parent.send(('reset_task', None))
        return np.stack([parent.recv() for parent in self.parents])
    
    def close(self):
        if self.closed:
            return
        if self.waiting:
            for parent in self.parents:
                parent.recv()
        for parent in self.parents:
            parent.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
        
        
        
                                          
        
        