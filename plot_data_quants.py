import matplotlib.pyplot as plt
from ogm.parser import TextParser
import os
import argparse as ap

# Get paramters via CLI
argparser = ap.ArgumentParser()
argparser.add_argument("datafile_name", help="Path to data table")
argparser.add_argument(
    "--date_format",
    help="Python datetime code to parse dates in the data",
    default="%Y-%m-%dT%H:%M:%SZ",
)
argparser.add_argument(
    "--date_key", required=True, help="Heading which contains timestamps in data"
)
argparser.add_argument(
    "--bucket_size", type=int, required=True, help="Size of each time interval in days"
)
argparser.add_argument(
    "--start_date",
    help="Leave empty to use earliest date in data; timestamp should be formatted as specified in date_format",
)
argparser.add_argument(
    "--end_date",
    help="Leave empty to use today; timestamp should be formatted as specified in date_format",
)
argparser.add_argument("--plot_title", default="Data Quantities", help="Title for plot")
args = argparser.parse_args()

# Generate plot
parser = TextParser()
parser.parse_file(args.datafile_name)
a = parser.plot_data_quantities(
    key=args.date_key,
    data_format=args.date_format,
    days_interval=args.bucket_size,
    start_date=args.start_date,
    end_date=args.end_date,
    plot_title=args.plot_title,
    show_plot=False,
)
plt.xticks(rotation="vertical")
plt.show()
