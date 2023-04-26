import pandas as pd

from . import load


class DataProcessor:
    def __init__(self):
        self.issue_data = None

    def process_data(self) -> pd.DataFrame:
        """
        This function adds new columns to the extracted data and returns processed dataframe

        return pd.DataFrame with processed data and added columns
        """
        self.extract_issues_data()

        # Drop columns with only NaN values
        self.issue_data.dropna(axis=1, how='all')

        # Add total days till resolution date:
        self._add_time_till_fix()

        self._process_timezones()

        # Sort columns on categorical / numerical datatype and alphabetically
        self._sort_columns()

        return self.issue_data

    def extract_issues_data(self) -> pd.DataFrame:
        """
        This function extracts data from json and csv files, and returns a dataframe

        return pd.DataFrame with extracted data from avro-issues.json
        """
        issues_json = load.issues_json()
        issues_csv = load.issues_csv()

        # create an empty dataframe
        issue_data = pd.DataFrame()

        # extract key column from issues_csv
        issue_data['key'] = issues_csv['key']

        # create empty lists for the columns of the history dataframe
        key_list_h = []
        id = []
        log_size = []
        created_log = []
        author = []
        author_active = []
        author_timezone = []
        to_status = []
        transition = []

        # create empty lists for the columns of the fields dataframe
        key_list_f = []
        last_viewed = []
        updated = []
        assignee = []
        duedate = []
        issue_type = []
        reporter = []
        reporter_timezone = []
        created = []
        priority = []
        resolution_date = []
        resolution_name = []
        watch_count = []
        comment_count = []
        summary_length = []
        description_length = []
        status = []

        # iterate over the issues_json list
        for i in range(len(issues_json)):
            data_key = issues_json[i]
            key = data_key['key']
            total_log = data_key['changelog']['total']
            history = data_key['changelog']['histories']
            fields = data_key['fields']

            # append data to the lists for fields dataframe
            key_list_f.append(key)
            last_viewed.append(fields['lastViewed'])
            updated.append(fields['updated'])
            assignee.append(fields['assignee']['name'] if fields['assignee'] else None)
            duedate.append(fields['duedate'])
            issue_type.append(fields['issuetype']['name'])
            reporter.append(fields['reporter']['name'])
            reporter_timezone.append(fields['reporter']['timeZone'])
            created.append(fields['created'])
            priority.append(fields['priority']['name'])
            resolution_date.append(fields['resolutiondate'] if fields['resolutiondate'] else None)
            resolution_name.append(fields['resolution']['name'] if fields['resolution'] else None)
            watch_count.append(fields['watches']['watchCount'])
            comment_count.append(fields['comment']['total'])
            summary_length.append(len(fields['summary'].split()) if fields['summary'] else None)
            description_length.append(len(fields['description'].split()) if fields['description'] else None)
            status.append(fields['status']['statusCategory']['name'])

            # iterate over the history list and append data to the lists
            for h in history:
                key_list_h.append(key)
                id.append(h['id'])
                log_size.append(total_log)
                created_log.append(h['created'])
                to_status.append(f"{h['items'][0]['toString']}")
                transition.append(f"{h['items'][0]['fromString']} to {h['items'][0]['toString']}")
                author.append(h['author']['name'])
                author_active.append(h['author']['active'])
                author_timezone.append(h['author']['timeZone'])

        # create a dictionary with the fields lists as values and column names as keys
        fields_data = {
            'key': key_list_f,
            'last_viewed': last_viewed,
            'updated': updated,
            'assignee': assignee,
            'duedate': duedate,
            'issue_type': issue_type,
            'reporter': reporter,
            'timezone_reporter': reporter_timezone,
            'created': created,
            'priority': priority,
            'resolution_date': resolution_date,
            'watch_count': watch_count,
            'comment_count': comment_count,
            'summary_lenght': summary_length,
            'description_length': description_length,
            'status': status
        }
        # create a dictionary with the fields lists as values and column names as keys
        history_data = {
            'key': key_list_h,
            'id': id,
            'log_size': log_size,
            'created_log': created_log,
            'author': author,
            'author_active': author_active,
            'timezone_author': author_timezone,
            'to_status': to_status,
            'transition': transition
        }

        # convert the dictionaries to a dataframe
        df_fields = pd.DataFrame(fields_data)
        df_history = pd.DataFrame(history_data)

        # merge the history and fields dataframe with the original dataframe
        self.issue_data = issue_data.merge(df_fields, on='key', how='left').merge(df_history, on='key', how='left')

        return self.issue_data

    def _add_time_till_fix(self) -> None:
        """
        This function takes two datetime strings as input and returns the time difference between them in a timedelta object.

        Parameters:
        start (str): The starting datetime string in the format 'YYYY-MM-DD HH:MM:SS'
        end (str): The ending datetime string in the format 'YYYY-MM-DD HH:MM:SS'

        Returns:
        timedelta: The time difference between the start and end datetime strings
        """
        # Format date columns to DateTime format
        self.issue_data['resolution_date'] = pd.to_datetime(self.issue_data['resolution_date'])
        self.issue_data['created'] = pd.to_datetime(self.issue_data['created'])
        self.issue_data['created_log'] = pd.to_datetime(self.issue_data['created_log'])
        self.issue_data['updated'] = pd.to_datetime(self.issue_data['updated'])

        # Create column with total numbers of days till fix
        self.issue_data['days_till_fix'] = (self.issue_data['resolution_date'] - self.issue_data['created']).dt.days

    def _process_timezones(self):
        self.issue_data['timezone_reporter'] = self.issue_data['timezone_reporter'].str.split('/').str[0]
        self.issue_data['timezone_author'] = self.issue_data['timezone_author'].str.split('/').str[0]
        self.issue_data['timezone_reporter'] = self.issue_data['timezone_reporter'].replace('etc')

        # Create a column to check if timezone reporter equal timezone author (who works on the log)
        self.issue_data['timezone_diff'] = self.issue_data['timezone_reporter'] == self.issue_data['timezone_author']
        self.issue_data['timezone_diff'] = self.issue_data['timezone_diff'].astype(object)

        # Drop timezone author
        self.issue_data = self.issue_data.drop('timezone_author', axis=1)

    def _sort_columns(self):
        # Select categorical columns
        cat_cols = self.issue_data.select_dtypes(include=['object']).columns

        # Select numerical columns
        num_cols = self.issue_data.select_dtypes(exclude=['object']).columns

        # Sort both column sets
        sorted_cats = self.issue_data[cat_cols].sort_index()
        sorted_nums = self.issue_data[num_cols].sort_index()

        # Concatenate both sets
        self.issue_data = pd.concat([sorted_cats, sorted_nums], axis=1)
