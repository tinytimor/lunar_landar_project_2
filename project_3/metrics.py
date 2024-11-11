def extract_metrics(info, episode, ep_reward, num_soups_made, layout, layout_avg, overall_avg, gamma, lambda_gae, clip_epsilon, update_every):
    if 'episode' in info and 'ep_game_stats' in info['episode']:
        ep_game_stats = info['episode']['ep_game_stats']

        # ----- Onion Metrics -----
        num_onion_pickups_agent0 = len(ep_game_stats['onion_pickup'][0])
        num_onion_pickups_agent1 = len(ep_game_stats['onion_pickup'][1])
        total_onion_pickups = num_onion_pickups_agent0 + num_onion_pickups_agent1

        num_useful_onion_pickups_agent0 = len(ep_game_stats['useful_onion_pickup'][0])
        num_useful_onion_pickups_agent1 = len(ep_game_stats['useful_onion_pickup'][1])
        total_useful_onion_pickups = num_useful_onion_pickups_agent0 + num_useful_onion_pickups_agent1

        num_onion_drops_agent0 = len(ep_game_stats['onion_drop'][0])
        num_onion_drops_agent1 = len(ep_game_stats['onion_drop'][1])
        total_onion_drops = num_onion_drops_agent0 + num_onion_drops_agent1

        num_useful_onion_drops_agent0 = len(ep_game_stats['useful_onion_drop'][0])
        num_useful_onion_drops_agent1 = len(ep_game_stats['useful_onion_drop'][1])
        total_useful_onion_drops = num_useful_onion_drops_agent0 + num_useful_onion_drops_agent1

        potting_onion_agent0 = len(ep_game_stats['potting_onion'][0])
        potting_onion_agent1 = len(ep_game_stats['potting_onion'][1])
        total_potting_onion = potting_onion_agent0 + potting_onion_agent1

        optimal_potting_onion_agent0 = len(ep_game_stats['optimal_onion_potting'][0])
        optimal_potting_onion_agent1 = len(ep_game_stats['optimal_onion_potting'][1])
        total_optimal_potting_onion = optimal_potting_onion_agent0 + optimal_potting_onion_agent1

        viable_potting_onion_agent0 = len(ep_game_stats['viable_onion_potting'][0])
        viable_potting_onion_agent1 = len(ep_game_stats['viable_onion_potting'][1])
        total_viable_potting_onion = viable_potting_onion_agent0 + viable_potting_onion_agent1

        catastrophic_potting_onion_agent0 = len(ep_game_stats['catastrophic_onion_potting'][0])
        catastrophic_potting_onion_agent1 = len(ep_game_stats['catastrophic_onion_potting'][1])
        total_catastrophic_potting_onion = catastrophic_potting_onion_agent0 + catastrophic_potting_onion_agent1

        useless_potting_onion_agent0 = len(ep_game_stats['useless_onion_potting'][0])
        useless_potting_onion_agent1 = len(ep_game_stats['useless_onion_potting'][1])
        total_useless_potting_onion = useless_potting_onion_agent0 + useless_potting_onion_agent1

        # ----- Dish Metrics -----
        num_dish_pickups_agent0 = len(ep_game_stats['dish_pickup'][0])
        num_dish_pickups_agent1 = len(ep_game_stats['dish_pickup'][1])
        total_dish_pickups = num_dish_pickups_agent0 + num_dish_pickups_agent1

        num_useful_dish_pickups_agent0 = len(ep_game_stats['useful_dish_pickup'][0])
        num_useful_dish_pickups_agent1 = len(ep_game_stats['useful_dish_pickup'][1])
        total_useful_dish_pickups = num_useful_dish_pickups_agent0 + num_useful_dish_pickups_agent1

        num_dish_drops_agent0 = len(ep_game_stats['dish_drop'][0])
        num_dish_drops_agent1 = len(ep_game_stats['dish_drop'][1])
        total_dish_drops = num_dish_drops_agent0 + num_dish_drops_agent1

        num_useful_dish_drops_agent0 = len(ep_game_stats['useful_dish_drop'][0])
        num_useful_dish_drops_agent1 = len(ep_game_stats['useful_dish_drop'][1])
        total_useful_dish_drops = num_useful_dish_drops_agent0 + num_useful_dish_drops_agent1

        # ----- Soup Metrics -----
        num_soup_pickups_agent0 = len(ep_game_stats['soup_pickup'][0])
        num_soup_pickups_agent1 = len(ep_game_stats['soup_pickup'][1])
        total_soup_pickups = num_soup_pickups_agent0 + num_soup_pickups_agent1

        num_soup_deliveries_agent0 = len(ep_game_stats['soup_delivery'][0])
        num_soup_deliveries_agent1 = len(ep_game_stats['soup_delivery'][1])
        total_soup_deliveries = num_soup_deliveries_agent0 + num_soup_deliveries_agent1

        incorrect_soup_delivery = ep_game_stats.get('incorrect_soup_delivery', [[], []])
        num_incorrect_deliveries_agent0 = len(incorrect_soup_delivery[0])
        num_incorrect_deliveries_agent1 = len(incorrect_soup_delivery[1])
        total_incorrect_deliveries = num_incorrect_deliveries_agent0 + num_incorrect_deliveries_agent1

        # ----- Reward Metrics -----
        cumulative_sparse_rewards_agent0 = ep_game_stats['cumulative_sparse_rewards_by_agent'][0]
        cumulative_sparse_rewards_agent1 = ep_game_stats['cumulative_sparse_rewards_by_agent'][1]
        total_cumulative_sparse_rewards = cumulative_sparse_rewards_agent0 + cumulative_sparse_rewards_agent1

        cumulative_shaped_rewards_agent0 = ep_game_stats['cumulative_shaped_rewards_by_agent'][0]
        cumulative_shaped_rewards_agent1 = ep_game_stats['cumulative_shaped_rewards_by_agent'][1]
        total_cumulative_shaped_rewards = cumulative_shaped_rewards_agent0 + cumulative_shaped_rewards_agent1

        potting_efficiency_onion_agent0 = (optimal_potting_onion_agent0 / potting_onion_agent0) if potting_onion_agent0 > 0 else 0
        potting_efficiency_onion_agent1 = (optimal_potting_onion_agent1 / potting_onion_agent1) if potting_onion_agent1 > 0 else 0
        average_potting_efficiency_onion = (potting_efficiency_onion_agent0 + potting_efficiency_onion_agent1) / 2

        # ----- Collaboration Metrics -----
        total_optimal_potting = total_optimal_potting_onion
        total_potting = total_potting_onion
        coordination_success_rate = (total_optimal_potting / total_potting) if total_potting > 0 else 0

        # Compile all metrics into a dictionary
        episode_data = {
            'episode': episode,
            'reward': ep_reward,
            'soups': num_soups_made,
            'layout': layout,
            'layout_average': layout_avg,
            'overall_average': overall_avg,
            'gamma': gamma,
            'lambda_gae': lambda_gae,
            'clip_epsilon': clip_epsilon,
            'update_every': update_every,
            'total_onion_pickups': total_onion_pickups,
            'total_useful_onion_pickups': total_useful_onion_pickups,
            'total_onion_drops': total_onion_drops,
            'total_useful_onion_drops': total_useful_onion_drops,
            'total_potting_onion': total_potting_onion,
            'total_optimal_potting_onion': total_optimal_potting_onion,
            'total_viable_potting_onion': total_viable_potting_onion,
            'total_catastrophic_potting_onion': total_catastrophic_potting_onion,
            'total_useless_potting_onion': total_useless_potting_onion,
            'total_dish_pickups': total_dish_pickups,
            'total_useful_dish_pickups': total_useful_dish_pickups,
            'total_dish_drops': total_dish_drops,
            'total_useful_dish_drops': total_useful_dish_drops,
            'total_soup_pickups': total_soup_pickups,
            'total_soup_deliveries': total_soup_deliveries,
            'total_incorrect_deliveries': total_incorrect_deliveries,
            'total_cumulative_sparse_rewards': total_cumulative_sparse_rewards,
            'total_cumulative_shaped_rewards': total_cumulative_shaped_rewards,
            'average_potting_efficiency_onion': average_potting_efficiency_onion,
            'coordination_success_rate': coordination_success_rate
        }
    else:
        # Default values if metrics are not available
        episode_data = {
            'episode': episode,
            'reward': ep_reward,
            'soups': num_soups_made,
            'layout': layout,
            'layout_average': layout_avg,
            'overall_average': overall_avg,
            'gamma': gamma,
            'lambda_gae': lambda_gae,
            'clip_epsilon': clip_epsilon,
            'update_every': update_every,
            'total_onion_pickups': 0,
            'total_useful_onion_pickups': 0,
            'total_onion_drops': 0,
            'total_useful_onion_drops': 0,
            'total_potting_onion': 0,
            'total_optimal_potting_onion': 0,
            'total_viable_potting_onion': 0,
            'total_catastrophic_potting_onion': 0,
            'total_useless_potting_onion': 0,
            'total_dish_pickups': 0,
            'total_useful_dish_pickups': 0,
            'total_dish_drops': 0,
            'total_useful_dish_drops': 0,
            'total_soup_pickups': 0,
            'total_soup_deliveries': 0,
            'total_incorrect_deliveries': 0,
            'total_cumulative_sparse_rewards': 0,
            'total_cumulative_shaped_rewards': 0,
            'average_onion_pickup_efficiency': 0,
            'average_potting_efficiency_onion': 0,
            'coordination_success_rate': 0
        }

    return episode_data