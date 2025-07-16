# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
"""This module defines the Quarter class."""

from datetime import date, timedelta
import evaltools as evt


class Quarter(object):
    """
    Year quarter.

    Class of objects defining year quarters (used in
    Evaluator.quarterlyMedianScore and plotting.plot_quarterlyMedianScore)
    Quarters can be cut in a classic way (the first quarter begins in January)
    or in the climatology way (the first quarter begins in March).

    """

    def __init__(self, start_date, end_date):
        """
        Quarter constructor.

        Parameters
        ----------
        start_date : datetime.date
            Start date of the quarter.
        end_date : datetime.date
            End date of the quarter.

        """
        # cutting specification
        if start_date.month in [3, 6, 9, 12]:
            cutting = 'climato'
        elif start_date.month in [1, 4, 7, 10]:
            cutting = 'classic'
        else:
            raise evt.EvaltoolsError(
                f"{start_date} - {end_date} is not a valid year quarter !!!"
            )

        # check if start_date and end_date are correct
        if (start_date.day != 1
                or (end_date + timedelta(days=1)).day != 1
                or end_date.month % 12 != (start_date.month+2) % 12
                or start_date.year < 2000
                or end_date.year < 2000
                or start_date.year >= 3000
                or end_date.year >= 3000
                or start_date.year not in [end_date.year, end_date.year-1]):
            raise evt.EvaltoolsError(
                f"{start_date} - {end_date} is not a valid year quarter !!!"
            )

        # set attributes
        self.start_date = start_date
        self.end_date = end_date

        year = str(self.end_date.year)

        trigram = {'3': 'MAM', '6': 'JJA', '9': 'SON', '12': 'DJF',
                   '1': 'JFM', '4': 'AMJ', '7': 'JAS', '10': 'OND'}
        months = trigram[str(start_date.month)]

        self.string = months + year
        self.cutting = cutting

    @classmethod
    def from_string(cls, string):
        """
        Construct a Quarter from its string representation.

        Parameters
        ----------
        string : str
            String representation of the Quarter to construct.

        """
        trigram = {'MAM': 3, 'JJA': 6, 'SON': 9, 'DJF': 12,
                   'JFM': 1, 'AMJ': 4, 'JAS': 7, 'OND': 10}

        months = string[:3]

        year = string[3:7]

        if int(year) < 2000 or int(year) >= 3000:
            raise evt.EvaltoolsError(
                f"{string} is not a valid year quarter !!!"
            )

        start_date = date(
            int(year) - 1*(months == 'DJF'),
            trigram[months],
            1,
        )
        end_date = date(
            int(year) + 1*(months == 'OND'),
            (trigram[months]+2) % 12 + 1,
            1,
        ) - timedelta(days=1)

        quarter = cls(start_date, end_date)
        return quarter

    def __repr__(self):
        """Repr magic method."""
        return self.string

    def __gt__(self, obj):
        """Greater than."""
        return self.start_date > obj.start_date

    def __ge__(self, obj):
        """Greater or equal."""
        return self.start_date >= obj.start_date

    def __eq__(self, obj):
        """Equal."""
        return self.start_date == obj.start_date

    def range(self, start_quarter):
        """
        Consecutive quarters list.

        Parameters
        ----------
        start_quarter: Quarter object
            First quarter to insert in the list.

        Returns
        -------
        list of quarter.Quarter
            List of consecutive quarters ranging from start_quarter to the
            current object.

        """
        # check if self and start_quarter have same format
        if self.cutting != start_quarter.cutting:
            raise evt.EvaltoolsError(
                f"{start_quarter} has a different cutting format than the "
                f"current Quarter object !!!"
            )

        month = start_quarter.start_date.month
        year = start_quarter.start_date.year
        res = []
        q = start_quarter
        while q < self:
            res.append(q)
            year = year + (month in [10, 11, 12])
            month = (month+2) % 12 + 1
            q = Quarter(date(year, month, 1),
                        date(year + (month in [10, 11, 12]),
                             (month+2) % 12 + 1,
                             1) - timedelta(days=1))
        res.append(q)
        return res
