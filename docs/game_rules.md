# Games and reported metric

The generators in `sequence_form/` are the Python games used by this
reproduction package. The named paper configurations are:

| Game | Configuration | Main-run iterations |
| --- | --- | ---: |
| Matrix games | 10 x 10, seeds 0, 1, 2, 3 | 2,000 |
| Kuhn poker | 3 ranks | 400 |
| Leduc poker | 3 ranks | 10,000 |
| Goofspiel | 4 cards | 2,000 |
| Liar's Dice | 1 die per player, 6 faces per die, no wild faces | 100 |

In the Liar's Dice generator, a challenge compares the bid quantity with the
number of dice showing exactly the bid face. There is no wild-face rule.
This differs from Liar's Dice configurations that treat the highest face as
wild. The historical paper data use the supplied Python game; the package
does not substitute a different game or the later OpenSpiel comparison runs.

The quantity called *Exploitability* in the figures is the sum of the two
players' unilateral best-response gains (NashConv), in original payoff units.
It is not divided by two. RT methods are evaluated using their current
strategies. Baseline configurations state whether the evaluated strategy is
the current strategy or the specified average.

Chance probabilities are included in the sequence-form payoff matrices.
The game builders and best-response evaluator are included so that reruns
use the same game representation and metric.
