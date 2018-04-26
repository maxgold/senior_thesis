def update_params(actor_critic, rollouts, optimizer, args):
    if args.algo in ['a2c', 'acktr']:
        values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(Variable(rollouts.observations[:-1].view(-1, *obs_shape)),
                                                                                       Variable(rollouts.states[0].view(-1, actor_critic.state_size)),
                                                                                       Variable(rollouts.masks[:-1].view(-1, 1)),
                                                                                       Variable(rollouts.actions.view(-1, action_shape)))

        values = values.view(args.num_steps, args.num_processes, 1)
        action_log_probs = action_log_probs.view(args.num_steps, args.num_processes, 1)

        advantages = Variable(rollouts.returns[:-1]) - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(Variable(advantages.data) * action_log_probs).mean()

        if args.algo == 'acktr' and optimizer.steps % optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = Variable(torch.randn(values.size()))
            if args.cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - Variable(sample_values.data)).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            optimizer.acc_stats = False

        optimizer.zero_grad()
        (value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef).backward()

        if args.algo == 'a2c':
            nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)

        optimizer.step()
    elif args.algo == 'ppo':
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for e in range(args.ppo_epoch):
            if args.recurrent_policy:
                data_generator = rollouts.recurrent_generator(advantages,
                                                        args.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(advantages,
                                                        args.num_mini_batch)

            for sample in data_generator:
                observations_batch, states_batch, actions_batch, \
                   return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(Variable(observations_batch),
                                                                                               Variable(states_batch),
                                                                                               Variable(masks_batch),
                                                                                               Variable(actions_batch))

                adv_targ = Variable(adv_targ)
                ratio = torch.exp(action_log_probs - Variable(old_action_log_probs_batch))
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

                value_loss = (Variable(return_batch) - values).pow(2).mean()

                optimizer.zero_grad()
                (value_loss + action_loss - dist_entropy * args.entropy_coef).backward()
                nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)
                optimizer.step()
