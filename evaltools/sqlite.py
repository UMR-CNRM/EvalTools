# Copyright (c) Météo France (2017-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info
"""
This module defines Store class.

This class is designed to store timeseries data in SQLite format.

"""
import sqlite3
import numpy as np
import pandas as pd
import evaltools as evt
import os
# from packaging import version


class Store(object):
    """
    Class designed for storing time series data in SQLite format.

    To be handle by this class, SQLite tables must have a time
    ('%Y-%m-%dT%H:%M') as unique value and columns must correspond to
    stations codes.

    """

    def __init__(
            self, file_path, table, time_key_name='dt',
            create_if_not_exist=True):
        """
        Store constructor.

        Parameters
        ----------
        file_path : str
            Path of the netcdf file to read. The file must exist (use the
            classmethod newfile to create one).
        table : str
            Name of the table to read in the sqlite file.
        time_key_name : str
            Unique id of the sqlite table corresponding to the time of the
        observations.

        """
        if not os.path.isfile(file_path):
            if not create_if_not_exist:
                raise evt.EvaltoolsError("File {} not found".format(file_path))
            print(
                (
                    "File {} not found. A new DB file is created."
                ).format(file_path)
            )

        # connect to database
        self.db = sqlite3.connect(
            os.path.join(file_path)
        )
        self.cursor = self.db.cursor()
        self.table = table
        self.time_key_name = time_key_name

        # check if table exist
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        )
        db_tables = self.cursor.fetchall()
        db_tables = [t[0] for t in db_tables]
        if table not in db_tables:
            if not create_if_not_exist:
                raise evt.EvaltoolsError(
                    "Table {} not found in {}".format(table, file_path)
                )
            print(
                (
                    "Table {} not found in file. A new table is created."
                ).format(table)
            )
            self.cursor.execute(
                """
                    CREATE TABLE IF NOT EXISTS {tab}(
                        {pk} CHAR(13) PRIMARY KEY UNIQUE
                    )
                """.format(pk=time_key_name, tab=table)
            )
            self.db.commit()
        else:
            self.cursor.execute("PRAGMA table_info('{}')".format(table))
            table_info = self.cursor.fetchall()
            primary_keys = [
                col[1] for col in table_info if col[5] == 1
            ]
            if primary_keys != [time_key_name]:
                raise evt.EvaltoolsError(
                    (
                        "Primary keys of table {} are {} but should be {} " +
                        "in view of time_key_name argument value."
                    ).format(table, primary_keys, [time_key_name])
                )

    def __del__(self):
        """Delete magic method."""
        if hasattr(self, 'db'):
            self.db.close()

    def __enter__(self):
        """Enter magic method."""
        return self

    def __exit__(self, type, value, traceback):
        """Exit magic method."""
        if hasattr(self, 'db'):
            self.db.close()

    @property
    def nsta(self):
        """Get the number of stations in the table."""
        self.cursor.execute("PRAGMA table_info('{}')".format(self.table))
        table_info = self.cursor.fetchall()
        return len(table_info) - 1

    @property
    def ntimes(self):
        """Get the number of times in the table."""
        self.cursor.execute(
            "SELECT {key} FROM {table} ORDER BY {key}".format(
                table=self.table,
                key=self.time_key_name,
            )
        )
        times = self.cursor.fetchall()
        return len(times)

    @property
    def times(self):
        """Get the sequence of times in the table."""
        self.cursor.execute(
            "SELECT {key} FROM {table} ORDER BY {key}".format(
                table=self.table,
                key=self.time_key_name,
            )
        )
        times = self.cursor.fetchall()
        times = pd.DatetimeIndex([col[0] for col in times])
        return times

    @property
    def stations(self):
        """Get the station list in the table."""
        self.cursor.execute("PRAGMA table_info('{}')".format(self.table))
        table_info = self.cursor.fetchall()
        stations = [col[1] for col in table_info]
        stations.remove(self.time_key_name)
        return stations

    def get_station_ids(self):
        """Return station codes found in the database table."""
        return self.stations

    def get_times(self):
        """Return times found in the sqlite table."""
        return self.times

    def add_stations(self, new_stations):
        """
        Add stations to the sqlite table.

        New station codes must not already be present.

        Parameters
        ----------
        new_stations : list of str
            List of the codes of the new stations.

        """
        sql_cmd = "ALTER TABLE {table} ADD COLUMN '{col}' {col_type}"
        for c in new_stations:
            self.cursor.execute(
                sql_cmd.format(
                    table=self.table,
                    col=c,
                    col_type="DECIMAL(5,2)",
                )
            )
        self.db.commit()

    def _remove_rows(self, condition):
        """
        Remove rows corresponding to the provided condition.

        Parameters
        ----------
        condition : str
            Condition to select removed rows.

        """
        self.cursor.execute(
            "DELETE FROM {} WHERE {}".format(self.table, condition)
        )
        self.db.commit()

    def remove_old_data(self, nb_days):
        """
        Remove rows older than the provided number of days.

        Parameters
        ----------
        nb_days : int
            Rows corresonding to data older than nb_days days are deleted.

        """
        self._remove_rows(
            "date({}) < date('now', '-{} days')".format(
                self.time_key_name,
                nb_days,
            )
        )

    def remove_empty_columns(self):
        """Remove columns containing only NaN values."""
        df = self.get_dataframe()
        idx = df.isna().all(axis=0)

        # # for sqlite version >= 3.35.0
        # if version.parse(sqlite3.sqlite_version) >= version.parse('3.35.0'):
        #     cols_to_rm = df.columns[idx]
        #     sql_cmd = "ALTER TABLE {table} DROP COLUMN '{col}'"
        #     for c in cols_to_rm:
        #         self.cursor.execute(
        #             sql_cmd.format(
        #                 table=self.table,
        #                 col=c,
        #             )
        #         )
        #     self.db.commit()
        # else:
        if idx.any():
            kept_cols = df.columns[~idx]
            self.cursor.execute(
                (
                    "CREATE TABLE {tab}_temp(\n" +
                    "    {pk} CHAR(13) PRIMARY KEY UNIQUE,\n" +
                    "    {cols}\n" +
                    ")"
                ).format(
                    pk=self.time_key_name,
                    tab=self.table,
                    cols=',\n    '.join(
                        [col + ' DECIMAL(5,2)' for col in kept_cols]
                    ),
                )
            )
            self.cursor.execute(
                "INSERT INTO {t}_temp({cols}) SELECT {cols} FROM {t} ".format(
                    t=self.table,
                    cols=", ".join(
                        [self.time_key_name] +
                        ['"{}"'.format(col) for col in kept_cols.to_list()]
                    ),
                )
            )
            self.cursor.execute("DROP TABLE {};".format(self.table))
            self.cursor.execute(
                "ALTER TABLE {t}_temp RENAME TO {t};".format(t=self.table)
            )
            self.db.commit()

    def get_dataframe(
            self, start_date=None, end_date=None, stations=None):
        """
        Get data contained within the database.

        Requested variable must be 2-dimensional: the first dimension
        corresponding to time and the second one to the diferent measurement
        sites.

        Parameters
        ----------
        name : str
            Name of the variable to retrieve.
        start_date : datetime.date object
            The date from which data is collected.
        end_date : datetime.date object
            The date until which data is collected.
        stations : None or list of str
            List of stations to keep in the returned dataset.
        step : int
            Time step in hours (ignored if series_type == 'daily').

        Returns
        -------
            evaltools.dataset.Dataset

        """
        if stations is None:
            selected_keys = '*'
            columns = [self.time_key_name] + self.stations
        else:
            selected_keys = [self.time_key_name]
            selected_keys.extend(list(stations))
            columns = list(selected_keys)
            selected_keys = ','.join(['"{}"'.format(k) for k in selected_keys])

        sql_cmd = "SELECT {keys} FROM {table}".format(
            table=self.table,
            keys=selected_keys,
        )

        # set period
        if start_date or end_date:
            sql_cmd += " WHERE "
            if start_date and end_date:
                sql_cmd += (
                    "date({}) >= date('{}') AND date({}) <= date('{}')"
                ).format(
                    self.time_key_name,
                    start_date.strftime("%Y-%m-%dT%H:%M"),
                    self.time_key_name,
                    end_date.strftime("%Y-%m-%dT%H:%M"),
                )
            elif start_date:
                sql_cmd += "date({}) >= date('{}')".format(
                    self.time_key_name,
                    start_date.strftime("%Y-%m-%dT%H:%M"),
                )
            elif end_date:
                sql_cmd += "date({}) <= date('{}')".format(
                    self.time_key_name,
                    end_date.strftime("%Y-%m-%dT%H:%M"),
                )

        sql_cmd += " ORDER BY {}".format(self.time_key_name)

        self.cursor.execute(sql_cmd)
        data = self.cursor.fetchall()

        df = pd.DataFrame(data, columns=columns)
        df.set_index(self.time_key_name, inplace=True)
        df.index = pd.DatetimeIndex(df.index)

        return df

    def get_dataset(
            self, dataset_name=None, start_date=None, end_date=None,
            stations=None, series_type='hourly', step=1):
        """
        Get data contained within the database.

        Requested variable must be 2-dimensional: the first dimension
        corresponding to time and the second one to the diferent measurement
        sites.

        Parameters
        ----------
        dataset_name : str
            Species name given to the return dataset.
        start_date : datetime.date object
            The date from which data is collected.
        end_date : datetime.date object
            The date until which data is collected.
        stations : None or list of str
            List of stations to keep in the returned dataset.
        series_type : str.
            It can be 'hourly' or 'daily'.
        step : int
            Time step in hours (ignored if series_type == 'daily').

        Returns
        -------
            evaltools.dataset.Dataset

        """
        df = self.get_dataframe(
            start_date=start_date,
            end_date=end_date,
            stations=stations,
        )

        # construct Dataset
        if dataset_name is None:
            species = self.table
        else:
            species = dataset_name

        start = (start_date if start_date else self.times.min().date())
        end = (end_date if end_date else self.times.max().date())
        if pd.isna(end) or pd.isna(start):
            print("Warning: could not infer period.")
            return df

        res = evt.dataset.Dataset(
            stations=df.columns,
            start_date=start,
            end_date=end,
            species=species,
            series_type=series_type,
            step=step,
        )
        res.data.update(df)

        return res

    def _update_table(self, pk_value, columns, values):
        """
        Update database table.

        Parameters
        ----------
        pk_value : str
            Value corresponding to the primary key columns.
        columns : list
            List of variables corresponding to new values list.
        values : list of scalar
            New values to add corresponding to given columns.

        """
        if len(columns) == 0:
            print("No data to update.")
            return
        if len(columns) != len(values):
            print("Warning: Columns length != values length.")
            return

        columns = ["'{}'".format(c) for c in columns]

        try:
            cols = ', '.join([self.time_key_name] + list(columns))
            vals = ["'{}'".format(pk_value)] + [str(val) for val in values]
            vals = ", ".join(vals)
            sql_cmd = "INSERT INTO {tab}({cols}) VALUES ({vals})"
            sql_cmd = sql_cmd.format(tab=self.table, cols=cols, vals=vals)
            self.cursor.execute(sql_cmd)
        except sqlite3.IntegrityError:  # in case unique id already exists
            tuples = ",".join(
                ["{}={}".format(c, v) for c, v in zip(columns, values)]
            )
            sql_cmd = (
                "UPDATE {tab} SET {tuples} WHERE {uniqueid}='{uniquevalue}'"
            )
            sql_cmd = sql_cmd.format(
                tab=self.table,
                tuples=tuples,
                uniqueid=self.time_key_name,
                uniquevalue=pk_value,
            )
            self.cursor.execute(sql_cmd)

        self.db.commit()

    def update_from_dataframe(
            self, dataframe, add_new_stations=True, nb_decimals=2):
        """
        Update database with a pandas.DataFrame object.

        Modify a table of the sqlite file using non-NA values from passed
        pandas.DataFrame object.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            DataFrame containing new values to add to the file.
        add_new_stations : bool
            If true, stations only present in dataset are also
            added to the sqlite file.
        nb_decimals : int
            Number of decimals to keep for the new values. Using less decimals
            will not decrease the size of the database files, but when
            archived with tar czf, these files can be much smaller.

        """
        # get station list from sqlite file and add some if missing
        if add_new_stations:
            new_stations = dataframe.columns[
                np.logical_not(np.in1d(dataframe.columns, self.stations))
            ]
            self.add_stations(new_stations)

        for dtime in dataframe.index:
            new_values = dataframe.loc[dtime]
            new_values = new_values.round(nb_decimals).replace(np.nan, 'NULL')
            self._update_table(
                pk_value=dtime.strftime("%Y-%m-%dT%H:%M"),
                columns=dataframe.columns,
                values=new_values,
            )
