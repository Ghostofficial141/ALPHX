from base64 import b64encode
from dotenv import load_dotenv, set_key
import os
import requests
import sys
from time import sleep
import traceback


erase_line = '\r\x1b[2K'


class API:
    base = 'https://api.worldquantbrain.com'
    auth = base + '/authentication'
    data_sets = base + '/data-sets'
    data_fields = base + '/data-fields'
    operators = base + '/operators'
    alphas = base + '/users/self/alphas'
    tutorials = base + '/tutorials'
    tutorial = base + '/tutorial-pages/'
    simul = base + '/simulations'
    alpha = base + '/alphas/'

    @staticmethod
    def performance(alpha_id):
        return f'{API.base}/competitions/{simul.Competition}/alphas/{alpha_id}/before-and-after-performance'

    @staticmethod
    def check_submission(alpha_id):
        return f'{API.base}/alphas/{alpha_id}/check'

    @staticmethod
    def pnl(alpha_id):
        return f'{API.base}/alphas/{alpha_id}/recordsets/pnl'


class data_set:
    D1 = ['model16', 'model51', 'model53', 'model77', 'option9', 'sentiment1']
    evergreen = ['analyst4', 'fundamental2', 'fundamental6', 'news12',
                 'news18', 'option8', 'pv1', 'pv13', 'socialmedia8', 'socialmedia12']

    analyst4 = "Analyst Estimate Data for Equity"
    fundamental2 = "Report Footnotes"
    fundamental6 = "Company Fundamental Data for Equity"
    news12 = "US News Data"
    news18 = "Ravenpack News Data"
    option8 = "Volatility Data"
    pv1 = "Price Volume Data for Equity"
    pv13 = "Relationship Data for Equity"
    socialmedia8 = "Social Media Data for Equity"
    socialmedia12 = "Sentiment Data for Equity"

    model16 = "Fundamental Scores"
    model51 = "Systematic Risk Metrics"
    model53 = "Creditworthiness Risk Measure Model"
    model77 = "Analysts' Factor Model"
    option9 = "Options Analytics"
    sentiment1 = "Research Sentiment Data"


class simul:
    Competition = 'IQC2025S2'

    Region = ['USA']
    Delay = [0, 1]
    Universe = ['TOP3000', 'TOP1000', 'TOP500', 'TOP200', 'TOPSP500']
    Neutralization = ['NONE', 'MARKET', 'SECTOR', 'INDUSTRY', 'SUBINDUSTRY']
    NaN_Handling = ['ON', 'OFF']


class clr:
    black = '\x1b[0;30m'
    red = '\x1b[0;31m'
    green = '\x1b[0;32m'
    yellow = '\x1b[0;33m'
    blue = '\x1b[0;34m'
    purple = '\x1b[0;35m'
    cyan = '\x1b[0;36m'
    white = '\x1b[0;37m'


load_dotenv()


def wq_login():
    t = os.getenv('t')

    wq_session = requests.Session()

    wq_session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Edge/79.0.1451.30 Safari/537.36',
        'Authorization': f'Bearer {t}'
    })
    wq_session.cookies.update({
        't': t
    })

    return wq_session


def submittable_alphas(alphas):
    """Return alphas where ALL checks (not just the first 6) have result == 'PASS'."""
    result = []

    for i in range(len(alphas)):
        checks = alphas[i].get('is', {}).get('checks', [])

        # Need at least 6 checks to consider this alpha
        if len(checks) < 6:
            continue

        is_submittable = True
        for check in checks:
            result_val = check.get('result', 'FAIL')
            # A check passes if its result is 'PASS' or informational (not a hard fail)
            if result_val not in ('PASS', 'PASS_WITH_INFO'):
                is_submittable = False
                break

        if is_submittable:
            result.append(alphas[i])

    print(f'{clr.green}{len(result)} Submittable Alphas Found...{clr.white}')
    return result


class Alpha:
    def simulate(wq_session, simulation_data):
        while True:
            try:
                simulation_response = wq_session.post(
                    API.simul,
                    json=simulation_data
                )

                if 'Location' not in simulation_response.headers:
                    print(f'{clr.red}Error: Location header missing from simulation response!{clr.white}')
                    print(f'{clr.yellow}Status Code: {simulation_response.status_code}{clr.white}')
                    print(f'{clr.yellow}Response Text: {simulation_response.text}{clr.white}')
                    if simulation_response.status_code == 401:
                        print(f'{clr.red}WARNING: Your WorldQuant authentication token "t" in .env may have expired or is invalid.{clr.white}')

                    # Do not retry client errors (like 400 Bad Request) indefinitely. Return the error.
                    if 400 <= simulation_response.status_code < 500 and simulation_response.status_code != 429:
                        return {"error": simulation_response.text, "status_code": simulation_response.status_code}

                    sleep(5)
                    continue

                simulation_progress_url = simulation_response.headers['Location']
                break

            except Exception:
                print(f'{clr.red}{traceback.format_exc()}{clr.white}')
                sleep(2)

        trial = 0

        while True:
            trial += 1

            simulation_progress = wq_session.get(simulation_progress_url)
            progress_data = simulation_progress.json()

            if (simulation_progress.headers.get('Retry-After', 0) == 0):
                if 'alpha' in progress_data:
                    alpha_id = progress_data['alpha']
                    print(f'{erase_line}{clr.yellow}Alpha ID: {alpha_id}{clr.white}', end='')
                    break
                else:
                    error_msg = progress_data.get('message', 'No alpha ID returned')
                    status = progress_data.get('status')
                    print(f'\n{clr.red}Simulation failed: {error_msg} (Status: {status}){clr.white}')
                    return {"error": error_msg, "status": status}

            progress_percent = 100 * progress_data.get('progress', 0.0)

            print(f'{erase_line}{clr.yellow}Attempt #{trial} | Simulation Progress: {progress_percent}%{clr.white}', end='')

            sleep(2 * float(simulation_progress.headers['Retry-After']))

        while True:
            try:
                alpha = wq_session.get(API.alpha + alpha_id)
                alpha = alpha.json()
                break
            except Exception:
                print(f'{clr.red}{traceback.format_exc()}{clr.white}')
                sleep(1)

        while True:
            performance_response = wq_session.get(API.performance(alpha_id))
            if (performance_response.text):
                try:
                    performance_comparison = performance_response.json()
                    break
                except Exception:
                    print(f'{clr.red}{traceback.format_exc()}{clr.white}')
                    sleep(1)

            print(f'{erase_line}{clr.yellow}Alpha ID: {alpha_id} | Getting Performance...{clr.white}', end='')

            sleep(float(performance_response.headers['Retry-After']))

        alpha['Performance Comparison'] = performance_comparison

        try:
            score_change = performance_comparison['score']['after'] - performance_comparison['score']['before']
        except KeyError:
            print(f'\n{clr.yellow}Warning: "score" key not found in performance comparison response. Keys: {list(performance_comparison.keys())}. Details: {performance_comparison}{clr.white}')
            score_change = 0.0

        alpha['Score Change'] = score_change

        print(f'{erase_line}{clr.yellow}Alpha ID: {alpha_id} | Performance: {score_change}{clr.white}')

        return alpha

    def get_performance(wq_session, alpha_id):
        while True:
            perf_resp = wq_session.get(API.performance(alpha_id))
            if (perf_resp.text):
                try:
                    perf = perf_resp.json()
                    break
                except Exception:
                    print(f'{clr.red}{traceback.format_exc()}{clr.white}')
                    sleep(1)

            print(f'{clr.yellow}Getting Performance...{clr.white}')

            sleep(float(perf_resp.headers['Retry-After']))

        performance = perf['score']['after'] - perf['score']['before']

        return performance

    @staticmethod
    def is_submittable(alpha):
        """Return True if ALL checks on the alpha result in PASS or PASS_WITH_INFO."""
        checks = alpha.get('is', {}).get('checks', [])
        if len(checks) < 6:
            return False
        for check in checks:
            result_val = check.get('result', 'FAIL')
            if result_val not in ('PASS', 'PASS_WITH_INFO'):
                return False
        return True

    def to_text(alpha):
        settings = alpha['settings']
        regular = alpha['regular']

        regular = regular.replace(' ', '').replace('\r\n', '').replace('\n', '')

        alpha_txt = f"""
Alpha Expression:
{regular}
Simulation Settings:
Region: {settings['region']}
Universe: {settings['universe']}
Delay: {settings['delay']}
Decay: {settings['decay']}
Neutralization: {settings['neutralization']}
Truncation: {settings['truncation']}
Pasteurization: {settings['pasteurization']}
NaN Handling: {settings['nanHandling']}
"""

        return alpha_txt.strip()

    def reverse(alpha_expression):

        if (';' in alpha_expression):
            return ';reverse('.join(alpha_expression.rsplit(';', 1)) + ')'
        else:
            return 'reverse(' + alpha_expression + ')'