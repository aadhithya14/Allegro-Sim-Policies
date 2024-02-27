from gym.envs.registration import register 


register(
	id='CupUnstacking-v1',
	entry_point='dexterous_env.cup_unstacking_env:CupUnstackingEnv',
	max_episode_steps=76,
)

register(
	id='BowlUnstacking-v1',
	entry_point='dexterous_env.bowl_unstacking_env:BowlUnstackingEnv',
	max_episode_steps=76,
)

register(
    id = 'PlierPicking-v1',
    entry_point='dexterous_env.plier_picking_env:PlierPicking',
	max_episode_steps=80,
)

register(
    id = 'CardFlipping-v1',
    entry_point='dexterous_env.card_flipping_env:CardFlipping',
	max_episode_steps=80,
)


register(
    id = 'CardTurning-v1',
    entry_point='dexterous_env.card_turning_env:CardTurning',
	max_episode_steps=80,
)

register(
    id = 'PegInsertion-v1',
    entry_point='dexterous_env.peg_insertion_env:PegInsertion',
	max_episode_steps=80,
)

register(
    id = 'MintOpening-v1',
    entry_point='dexterous_env.mint_opening_env:MintOpening',
    max_episode_steps=80
)

register(
    id = 'MultiTask-v1',
    entry_point='dexterous_env.multi_task_env:MultiTask',
    max_episode_steps=240
)

register(
    id = 'SpongeFlippingSim-v1',
    entry_point='dexterous_env.sponge_flipping_sim_env:SpongeFlippingSim',
    max_episode_steps=80
)

register(
    id = 'SpongeFlippingSimMovingHand-v1',
    entry_point='dexterous_env.sponge_flipping_sim_moving_hand_env:SpongeFlippingSimMovingHand',
    max_episode_steps=80
)

register(
    id = 'CubeFlippingSim-v1',
    entry_point='dexterous_env.cube_flipping_sim_env:CubeFlippingSim',
    max_episode_steps=80
)

register(
    id = 'CubePinchGraspingSim-v1',
    entry_point='dexterous_env.pinch_grasping_sim_env:PinchGraspingSim',
    max_episode_steps=80
)

register(
    id= 'SawyerAllegro-v1',
    entry_point='dexterous_env.sawyer_allegro_env:SawyerAllegro',
    max_episode_steps=80
)

register(
    id = 'ObjectPicking-v1',
    entry_point='dexterous_env.object_picking_env:ObjectPicking',
    max_episode_steps=80
)