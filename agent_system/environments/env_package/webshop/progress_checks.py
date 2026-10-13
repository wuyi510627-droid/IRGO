"""WebShop-specific position hints. No reward override or success-probability claim."""

import copy


def inspect_visible_progress(env, inspect_search_results=False):
    from web_agent_site.envs.web_agent_text_env import get_reward

    server = env.unwrapped.server
    state = server.user_sessions[env.unwrapped.session]
    # 读取训练环境内部目标与评分分解；不是纯可见文本，也不是通用成功概率。
    goal = copy.deepcopy(state["goal"])
    if isinstance(goal["goal_options"], dict):
        goal["goal_options"] = list(goal["goal_options"].values())

    def details(asin, options):
        _, d = get_reward(
            server.product_item_dict[asin], goal, server.product_prices[asin], options, verbose=True
        )
        return dict(
            static_ok=d["r_type"] == 1.0 and d["r_att"] == 1.0 and bool(d.get("r_price", True)),
            option_fraction=float(d.get("r_option", 1.0)),
        )

    asin = state.get("asin")
    result = dict(asin=asin, static_ok=False, option_fraction=0.0)
    if asin:
        result.update(details(asin, state["options"]))
    if inspect_search_results:
        visible = {
            str(a).upper()
            for a in env.get_available_actions()["clickables"]
            if str(a).upper() in server.product_item_dict
        }
        result.update(
            visible_products=len(visible),
            visible_static_feasible=sum(details(a, {})["static_ok"] for a in visible),
        )
    return result
