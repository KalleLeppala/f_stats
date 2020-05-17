#   @copyright: 2019 by Kalle Leppälä
#   @license: MIT <http://www.opensource.org/licenses/mit-license.php>

import f_stats
from f_stats import (
    estimator_covariance,
    sample,
    jackknife,
    frequencies,
    f2_basis,
    f2_all,
    f3_all,
    f4_basis,
    f4_all,
    index)

README =  "# f_stats" + "\n"
README +=  f_stats.__doc__
README += "\n## "  + estimator_covariance.__doc__
README += "\n## "  + sample.__doc__
README += "\n## "  + jackknife.__doc__
README += "\n## "  + frequencies.__doc__
README += "\n## "  + f2_basis.__doc__
README += "\n## "  + f2_all.__doc__
README += "\n## "  + f3_all.__doc__
README += "\n## "  + f4_basis.__doc__
README += "\n## "  + f4_all.__doc__
README += "\n## "  + index.__doc__





with open('README.md', 'wt') as readme_file:
    readme_file.write(README)