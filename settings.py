class argumentsRL():
    def __init__(self):
        self.discrete         = True
        self.seed             = 1
        self.random_seed      = 1
        self.num_processes    = 8
        self.num_stack        = 1 # number of frames to stack
        self.render           = False
        self.render_env       = self.render
        self.write_summary    = False
        self.summary_dir      = None
        self.log_interval     = 5
        self.use_gym_monitor  = False
        self.monitor_dir      = None

        self.lr               = .01
        self.actor_lr         = .01
        self.critic_lr        = .01
        self.gamma            = .99
        self.tau              = .95
        self.value_loss_coef  = .5
        self.entropy_coef     = .01
        self.clip_grad_norm   = False
        self.max_grad_norm    = .5

        self.num_steps        = 500
        self.num_updates      = 1000
        self.max_episodes     = 100
        self.max_episode_len  = 1000
        self.mb_size          = 1
        self.buffer_size      = 100
   
        self.action_layers    = [5, 5, 5]
        self.action_dim       = 4
        self.value_layers     = [10, 1]
        self.obs_dim          = 4



