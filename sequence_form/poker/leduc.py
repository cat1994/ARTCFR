from sequence_form.extensive_form_game import extensive_form_game as efg
from scipy.sparse import lil_matrix


def init_efg(num_ranks=3,
             prox_infoset_weights=False,
             prox_scalar=-1,
             integer=False,
             all_negative=False,
             num_raise_sizes=1,
             max_bets=2):
    assert num_ranks >= 2
    deck_size = num_ranks * 2
    if integer:
        hand_combinations = deck_size - 2
        rollout_combinations = 1
    else:
        hand_combinations = 1.0 / float(deck_size * (deck_size - 1))
        rollout_combinations = 1.0 / float(deck_size * (deck_size - 1) * \
                                     (deck_size - 2))

    parent = ([], [])
    begin = ([], [])
    end = ([], [])
    payoff = []
    reach = []
    next_s = [1, 1]
    # below only used for outputting a payoff-shifted constant-sum game when
    # all_negative is True
    payoff_shift = 0
    if all_negative:
        payoff_shift = -15
        payoff_p1 = []

    def _p_chance(rnd, board, i, j):
        if rnd == 0:
            if i == j:
                return 2 * hand_combinations
            return 4 * hand_combinations
        if i == board and j == board:
            return 0  # =0， 算到达概率时避免除以0
        elif i == board or j == board or i == j:
            return 4 * rollout_combinations
        return 8 * rollout_combinations

    def _build_terminal(rnd, board, value, previous_seq):
        for i in range(num_ranks):
            for j in range(num_ranks):
                payoff.append((previous_seq[0][i], previous_seq[1][j],
                               _p_chance(rnd, board, i, j) *
                               (value(i, j) + payoff_shift)))
                if all_negative:
                    payoff_p1.append((
                        previous_seq[0][i], previous_seq[1][j],
                        _p_chance(rnd, board, i, j) * (-value(i, j) + \
                                                     payoff_shift)))

    def _build_fold(rnd, board, who_folded, win_amount, previous_seq):
        if who_folded == 1:
            win_amount = -win_amount

        def _value(i, j):
            return win_amount

        _build_terminal(rnd, board, _value, previous_seq)

    def _build_showdown(rnd, board, win_amount, previous_seq):
        def _value(i, j):
            if i == board:
                return -win_amount
            elif j == board:
                return win_amount
            elif i > j:
                return -win_amount
            elif j > i:
                return win_amount
            return 0

        _build_terminal(rnd, board, _value, previous_seq)

    def _build(rnd, board, actor, num_bets, pot, previous_seq):
        opponent = 1 - actor
        facing = pot[opponent] - pot[actor]  # face actual payoff
        pot_actor = pot[actor]
        num_actions = (facing >
                       0) + 1 + (num_bets < max_bets) * num_raise_sizes  # fold(if other ), call, raise
        action = 0
        first_action = (actor == 0 and num_bets == 0)

        info_set = len(begin[actor])
        for i in range(num_ranks):  # update info set information ()
            parent[actor].append(previous_seq[actor][i])
            begin[actor].append(next_s[actor])
            next_s[actor] += num_actions
            end[actor].append(next_s[actor])
            for j in range(num_ranks):
                reach.append((actor, info_set + i, previous_seq[opponent][j],
                              _p_chance(rnd, board, i, j)))

        def _pn(idx):  # update the information precious seq idx, which seq reaches the information set
            t = [begin[actor][info_set + i] + idx for i in range(num_ranks)]
            if actor == 0:
                return (t, previous_seq[1])
            return (previous_seq[0], t)

        if facing > 0:  # fold
            _build_fold(rnd, board, actor, pot[actor], _pn(action))
            action += 1

        pot[actor] = pot[opponent]  # update pot for next
        if first_action:  # check
            _build(rnd, board, opponent, 0, pot, _pn(action))
        elif rnd + 1 < 2:  # call and deal board card
            for board in range(num_ranks):
                _build(rnd + 1, board, 0, 0, pot, _pn(action))
        else:  # showdown
            _build_showdown(rnd, board, pot[actor], _pn(action))
        action += 1

        if num_bets < max_bets:
            if rnd == 0:
                init_raise_size = 2
            else:
                init_raise_size = 4
            for raise_amt in [
                    init_raise_size * raise_size
                    for raise_size in range(1, num_raise_sizes + 1)
            ]:
                pot[actor] = pot[opponent] + raise_amt
                _build(rnd, board, opponent, 1 + num_bets, pot, _pn(action))
                action += 1

        pot[actor] = pot_actor

    previous_seq = ([0] * num_ranks, [0] * num_ranks)
    _build(0, -1, 0, 0, [1, 1], previous_seq)

    if integer:
        payoff_matrix = lil_matrix((next_s[0], next_s[1]), dtype=int)
    else:
        payoff_matrix = lil_matrix((next_s[0], next_s[1]))
    for i, j, payoff_value in payoff:
        payoff_matrix[i, j] += payoff_value
    reach_matrix = (lil_matrix((len(begin[0]), next_s[1])), lil_matrix(
        (len(begin[1]), next_s[0])))
    for player, infoset, opponent_seq, prob in reach:
        reach_matrix[player][infoset, opponent_seq] += prob
    if all_negative:
        if integer:
            payoff_p1_matrix = lil_matrix((next_s[0], next_s[1]), dtype=int)
        else:
            payoff_p1_matrix = lil_matrix((next_s[0], next_s[1]))
        for i, j, payoff_value in payoff_p1:
            payoff_p1_matrix[i, j] += payoff_value
        return efg.ExtensiveFormGame(
            'Leduc-%d' % num_ranks,
            payoff_matrix,
            begin,
            end,
            parent,
            prox_infoset_weights=prox_infoset_weights,
            prox_scalar=prox_scalar,
            reach=reach_matrix,
            B=payoff_p1_matrix,
            offset=2 * payoff_shift * (deck_size * (deck_size - 1) *
                                       (deck_size - 2)))
    else:
        return efg.ExtensiveFormGame(  # default
            'Leduc-%d' % num_ranks,
            payoff_matrix,
            begin,
            end,
            parent,
            prox_infoset_weights=prox_infoset_weights,
            prox_scalar=prox_scalar,
            reach=reach_matrix)