from distutils.command import upload
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time, os, re, pandas as pd
from ...tools.informative_tb import informative_exception
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
from ...config.logging import get_logger
import warnings

logger = get_logger()
"""Logger for this module."""

WAIT_TIMEOUT = 90


class KinLibScraper:
    def __init__(self, default_dir: str, upload_path: str):
        self.default_dir = os.path.abspath(default_dir)
        self.upload_path = os.path.abspath(upload_path)
        self.download_path = self.default_dir
        assert os.path.exists(self.default_dir), f"{self.default_dir} does not exist."
        assert os.path.exists(self.upload_path), f"{self.upload_path} does not exist."

        options = Options()
        options.add_experimental_option(
            "prefs",
            {
                "download.default_directory": self.download_path,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": True,
                "detach": True,
            },
        )

        userAget = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0"
            " Safari/537.36"
        )
        options.add_argument("--user-agent=" + userAget)

        # for file in os.listdir(default_dir):
        #     os.unlink(default_dir + file)

        self.url = "https://kinase-library.phosphosite.org/sites"
        self.file_path = os.path.join(default_dir, "Targets.csv")
        self.driver = webdriver.Chrome(options=options)
        self.waiter = WebDriverWait(self.driver, WAIT_TIMEOUT)

    @staticmethod
    def get_dict_from_scrape(scraped_dir_path, selected_kin) -> dict[str, dict[str, float]]:
        ### Assume we have already run scrape.py
        mapping = {}
        for file in [x for x in os.listdir(scraped_dir_path) if x.endswith(".tsv")]:
            site_name = ""
            tab = None
            try:
                tab = pd.read_csv(os.path.join(scraped_dir_path, file), sep="\t")
                site_name = re.sub(r"-[0-9]+`[0-9]+\.tsv", "", file)
            except Exception as e:
                print(e)
                print("Problem:", file)

            mapping[site_name] = tab

        for k, v in mapping.items():
            mapping[k] = v.set_index("kinase").to_dict()["site_percentile"]

        selected_kin_dict = {}
        for site, dict_ in mapping.items():
            if selected_kin in dict_:
                selected_kin_dict[site] = dict_[selected_kin]
            else:
                logger.info(f'{selected_kin} not found for site {site} (Atlas cannot handle "Y" Sites.).')
                logger.warn("↳")

        return selected_kin_dict

    def _get_to_result_stage(self):
        # Navigate to the URL
        self.driver.get(self.url)
        # Find the file upload button on the form and send the file path

        logger.status("About to click upload")
        upload_field_default = self.driver.find_element(
            By.XPATH,
            "/html/body/app-root/app-content-wrapper/main/app-score-site/div[2]/div[1]/div/app-file-upload/label/input",
        )
        # Make upload_field_default not hidden
        self.driver.execute_script("arguments[0].style.visibility = 'visible';", upload_field_default)

        upload_field_default.send_keys(self.upload_path)
        logger.status("Upload button clicked")

        # Wait for upload to finish based on existence of confirmation text

        locator_info = (
            By.XPATH,
            "/html/body/app-root/app-content-wrapper/main/app-score-site/div[2]/div[1]/div/app-file-upload/p",
        )
        logger.status("Waiting upload to complete")
        self.waiter.until(EC.presence_of_element_located(locator_info))
        EC.text_to_be_present_in_element_attribute(locator_info, "innerHTML", "Uploading file: in progress (100 %)")

        logger.status("Upload complete; about to click submit; waiting for button to show.")

        # Find the submit button on the form and click it
        self.waiter.until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "/html/body/app-root/app-content-wrapper/main/app-score-site/div[2]/div[2]/div[2]/button",
                )
            )
        )
        submit_button = self.driver.find_element(
            By.XPATH, "/html/body/app-root/app-content-wrapper/main/app-score-site/div[2]/div[2]/div[2]/button"
        )
        submit_button.click()
        logger.status("Submit button clicked")

        # Wait for the results to load
        waiter = WebDriverWait(self.driver, WAIT_TIMEOUT)
        waiter.until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    "/html/body/app-root/app-content-wrapper/main/app-score-site/app-sites-result/div/div/div[1]/div/div[1]/label",
                )
            )
        )

        logger.status("Results visible")

    def scrape(self):
        self._get_to_result_stage()
        self._get_tables()

    def _get_max_page(self):
        max_page = 0
        for x in self.driver.find_element(
            By.XPATH,
            "/html/body/app-root/app-content-wrapper/main/app-score-site/app-sites-result/div/div/div[1]/div/div[2]/span",
        ).find_elements(By.XPATH, ".//*"):
            elt = x.get_attribute("innerHTML")
            assert isinstance(elt, str)
            if elt.isnumeric():
                max_page = int(elt)
        return max_page + 1

    def _get_max_tab(self):
        max_tab = 0
        max_tab += len(
            self.driver.find_element(
                By.XPATH,
                "/html/body/app-root/app-content-wrapper/main/app-score-site/app-sites-result/div/div/div[1]/div/table/tbody",
            ).find_elements(By.TAG_NAME, "tr")
        )
        return max_tab + 1

    def _get_tables(self):
        # For each page of results...
        try:
            max_page = self._get_max_page()
            for p in range(1, max_page):
                # For each table of results, scrape the table
                self.driver.execute_script("window.scrollTo(0, 0)")
                max_tab = self._get_max_tab()
                for i in range(1, max_tab):
                    logger.status(f"Starting download (Page {p}/{max_page - 1}, Table {i}/{max_tab - 1})...")
                    self.driver.execute_script("window.scrollTo(0, 0)")
                    # Find the next button and click it
                    tab_button = self.driver.find_element(
                        By.XPATH,
                        f"/html/body/app-root/app-content-wrapper/main/app-score-site/app-sites-result/div/div/div[1]/div/table/tbody/tr[{i}]",
                    )

                    which_short_site_ls = []
                    for s in range(1, 4):
                        X = self.driver.find_element(
                            By.XPATH,
                            f"/html/body/app-root/app-content-wrapper/main/app-score-site/app-sites-result/div/div/div[1]/div/table/tbody/tr[{i}]/td/span[{s}]",
                        ).get_attribute("innerHTML")
                        which_short_site_ls.append(X)

                    which_short_site = "".join(which_short_site_ls)

                    tab_button.click()
                    # Find the download button and click it
                    self.waiter.until(EC.presence_of_element_located((By.XPATH, '//*[@id="downloadScoreSiteButton"]')))
                    download_button = self.driver.find_element(By.XPATH, '//*[@id="downloadScoreSiteButton"]')
                    download_button.click()
                    while (
                        len(
                            the_file_s := [
                                file
                                for file in os.listdir(os.path.expanduser(os.path.expandvars(self.download_path)))
                                if re.search(r"score-site-result-table\s*\(*[0-9]*\)*\.tsv", file)
                            ]
                        )
                        < 1
                    ):
                        time.sleep(0.01)
                    os.rename(
                        os.path.expanduser(os.path.expandvars(os.path.join(self.download_path, the_file_s[0]))),
                        new_name := os.path.expanduser(
                            os.path.expandvars(os.path.join(self.download_path, f"{which_short_site}-{p}`{i}.tsv"))
                        ),
                    )
                    while os.path.getsize(new_name) < 20_000 and os.path.split(new_name)[-1][5] != "Y":
                        time.sleep(0.01)

                    # except next button not found error
                # Find the next pagination button and click it
                if p != max_page - 1:
                    next_button = self.driver.find_element(By.LINK_TEXT, f"{p + 1}")
                    actions = ActionChains(self.driver)
                    actions.move_to_element(next_button).perform()
                    self.waiter.until(EC.element_to_be_clickable(next_button)).click()
        # Import NoSuchElementException from selenium.common.exceptions
        except Exception as e:
            informative_exception(e)


if __name__ == "__main__":  # pragma: no cover
    scraper = KinLibScraper(
        default_dir="/Users/druc594/Library/CloudStorage/OneDrive-PNNL/Desktop/ScrapeIntermediatesPLK/",
        upload_path="/Users/druc594/Library/CloudStorage/OneDrive-PNNL/Desktop/plk1sites.txt",
    )

    scraper.scrape()
