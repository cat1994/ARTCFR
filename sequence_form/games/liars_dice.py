from sequence_form.extensive_form_game import extensive_form_game as efg
from scipy.sparse import lil_matrix

from itertools import product



def init_efg(num_ranks=3,
             prox_infoset_weights=False,
             prox_scalar=-1,
             integer=False,
             all_negative=False,
             num_raise_sizes=1,
             max_bets=2):
    num_dice = 1 # every hold x dices
    
    deck_size = num_dice * 2

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



    def _generate_hands(num_dice, num_ranks):
        hc_dict = {}

        for i in product(range(num_ranks), repeat=num_dice):
            alpha = (1 / num_ranks) ** num_dice
            h = [0] * num_ranks
            for j in range(num_ranks):
                h[j] = i.count(j)

            str_h = ''  # only support num_rank<=9
            for i in h:
                str_h += str(i)
            if str_h not in hc_dict.keys():
                hc_dict[str_h] = alpha
            else:
                hc_dict[str_h] += alpha
        return hc_dict


    def _build_showdown(actor, bit_num_dice, bit_num_rank, previous_seq):
        def _challenge(h1, h2):
            # return challenge player loss result, if win return -1
            # count dice num and face num
            # hands type: [str, str]
            h1 = [int(i) for i in h1]
            h2 = [int(i) for i in h2]
            sign = -1 if actor==0 else 1
            # for loss
            # challenger win: the opponent liar
            # valid: the bid_num_dice is more than num of the same rank dice in desk
            if bit_num_dice > h1[bit_num_rank - 1] + h2[bit_num_rank - 1]:
                return 1.0*sign
            else:
                return -1.0*sign

        _build_terminal(_challenge, previous_seq)
    def _build_terminal(value, previous_seq):
        for i in range(size_info):
            for j in range(size_info):
                payoff.append((previous_seq[0][i], previous_seq[1][j],
                               _p_chance(i, j) *
                               (value(hands_list[i], hands_list[j]) + payoff_shift)))
                if all_negative:
                    payoff_p1.append((
                        previous_seq[0][i], previous_seq[1][j],
                        _p_chance(i, j) * (-value(hands_list[i], hands_list[j]) + \
                                                     payoff_shift)))
    def _p_chance(i,j):
        return hands_chance_dict[hands_list[i]]*hands_chance_dict[hands_list[j]]


    def _build(rnd, bit_num_dice, bit_num_rank, actor, previous_seq, ):



        opponent = 1 - actor
        if actor==0 and rnd==0:
            num_actions = num_dice*2 * num_ranks # 12
        else:
            num_actions = num_ranks - bit_num_rank + num_ranks*(num_dice*2-bit_num_dice) + 1
        action = 0
        first_action = actor == 0 and rnd == 0

        info_set = len(begin[actor])
        for i in range(size_info):
            if next_s[actor]==0 and actor==1:
                test=True
            parent[actor].append(previous_seq[actor][i])
            begin[actor].append(next_s[actor])
            next_s[actor] += num_actions
            end[actor].append(next_s[actor])
            for j in range(size_info):
                reach.append((actor, info_set + i, previous_seq[opponent][j],
                              _p_chance(i, j)))

        def _pn(idx):
            t = [begin[actor][info_set + i] + idx for i in range(size_info)]
            if actor == 0:
                return (t, previous_seq[1])
            return (previous_seq[0], t)

        if len(begin[0])>=10:
            pass
        if first_action:
            # Enumerate all opening bids for player 1.
            while action < num_actions:
                bit_num_dice = int(action / num_ranks) + 1
                bit_num_rank = action % num_ranks + 1
                _build(rnd + 1, bit_num_dice, bit_num_rank, opponent, _pn(action))
                action+=1

        else:
            while action<num_actions:

                if action == num_actions - 1:

                    _build_showdown(actor, bit_num_dice, bit_num_rank, _pn(action))
                elif action < num_ranks - bit_num_rank:  # increase num rank
                    r = bit_num_rank + action + 1

                    _build(rnd + 1, bit_num_dice, r, opponent, _pn(action))
                else:
                    # increase num rank
                    r=(action-(num_ranks-bit_num_rank))%num_ranks+1
                    d=bit_num_dice+int((action-(num_ranks-bit_num_rank))/num_ranks)+1

                    _build(rnd + 1, d, r, opponent, _pn(action))

                action += 1

    hands_chance_dict = _generate_hands(num_dice, num_ranks)
    hands_list = list(hands_chance_dict.keys())
    size_info = len(hands_list)
    previous_seq = ([0] * size_info, [0] * size_info)

    _build(0, 0, 0, 0, previous_seq)

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
            'Liars Dice-%d' % num_dice,
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
        return efg.ExtensiveFormGame(
            'Liars Dice-%d' % num_dice,
            payoff_matrix,
            begin,
            end,
            parent,
            prox_infoset_weights=prox_infoset_weights,
            prox_scalar=prox_scalar,
            reach=reach_matrix)
