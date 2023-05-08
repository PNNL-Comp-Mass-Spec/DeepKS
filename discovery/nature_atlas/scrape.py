from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time, os, re
from ...tools.informative_tb import informative_exception
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options

WAIT_TIMEOUT = 30


def main():
    options = Options()
    dd = r"/Users/druc594/Downloads/HIPK2-sites/"
    options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": dd,
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

    for file in os.listdir(dd):
        os.unlink(dd + file)

    # Replace this with the URL of the page you want to use
    url = "https://kinase-library.phosphosite.org/sites"

    # Replace this with the path to the file you want to upload
    file_path = "/Users/druc594/Library/CloudStorage/OneDrive-PNNL/Desktop/DeepKS_/DeepKS/discovery/nature_atlas/hipk2_sites_for_kin_lib_64.txt"

    # Replace this with the path where you want to download the results
    download_path = dd

    # Set up the web driver (in this case, for Chrome)
    driver = webdriver.Chrome(options=options)

    # Navigate to the URL
    driver.get(url)
    # Find the file upload button on the form and send the file path

    # for _ in range(5):
    #     driver.find_element(By.TAG_NAME, "html").send_keys(Keys.COMMAND, Keys.SUBTRACT, Keys.NULL) # TODO: Ubuntu version of this?

    print("About to click upload...")
    upload_field_default = driver.find_element(
        By.XPATH,
        "/html/body/app-root/app-content-wrapper/main/app-score-site/div[2]/div[1]/div/app-file-upload/label/input",
    )
    # Make upload_field_default not hidden
    driver.execute_script("arguments[0].style.visibility = 'visible';", upload_field_default)

    upload_field_default.send_keys(file_path)
    print("Upload button clicked...")

    # Wait for upload to finish based on existence of confirmation text
    wait = WebDriverWait(driver, WAIT_TIMEOUT)
    wait.until(
        EC.presence_of_element_located(
            (
                By.XPATH,
                "/html/body/app-root/app-content-wrapper/main/app-score-site/div[2]/div[1]/div/app-file-upload/p",
            )
        )
    )

    print("Upload complete; about to click submit...")

    # Find the submit button on the form and click it
    submit_button = driver.find_element(
        By.XPATH, "/html/body/app-root/app-content-wrapper/main/app-score-site/div[2]/div[2]/div[2]/button"
    )
    submit_button.click()
    print("Submit button clicked...")

    # Wait for the results to load
    wait = WebDriverWait(driver, WAIT_TIMEOUT)
    wait.until(
        EC.presence_of_element_located(
            (
                By.XPATH,
                "/html/body/app-root/app-content-wrapper/main/app-score-site/app-sites-result/div/div/div[1]/div/div[1]/label",
            )
        )
    )

    print("Results visible...")
    # For each page of results...
    try:
        for p in range(
            1,
            max_page := 1
            + max(
                [
                    int(x.get_attribute("innerHTML"))
                    for x in driver.find_element(
                        By.XPATH,
                        "/html/body/app-root/app-content-wrapper/main/app-score-site/app-sites-result/div/div/div[1]/div/div[2]/span",
                    ).find_elements(By.XPATH, ".//*")
                    if x.get_attribute("innerHTML").isnumeric()
                ]
            ),
        ):
            # For each table of results, scrape the table
            driver.execute_script("window.scrollTo(0, 0)")
            for i in range(
                1,
                max_tab := 1
                + len(
                    driver.find_element(
                        By.XPATH,
                        "/html/body/app-root/app-content-wrapper/main/app-score-site/app-sites-result/div/div/div[1]/div/table/tbody",
                    ).find_elements(By.TAG_NAME, "tr")
                ),
            ):
                print(f"Starting download (Page {p}/{max_page - 1}, Table {i}/{max_tab - 1})...")
                driver.execute_script("window.scrollTo(0, 0)")
                # Find the next button and click it
                tab_bttn = driver.find_element(
                    By.XPATH,
                    f"/html/body/app-root/app-content-wrapper/main/app-score-site/app-sites-result/div/div/div[1]/div/table/tbody/tr[{i}]",
                )
                which_short_site = "".join(
                    [
                        driver.find_element(
                            By.XPATH,
                            f"/html/body/app-root/app-content-wrapper/main/app-score-site/app-sites-result/div/div/div[1]/div/table/tbody/tr[{i}]/td/span[{s}]",
                        ).get_attribute("innerHTML")
                        for s in [1, 2, 3]
                    ]
                )
                tab_bttn.click()
                # Find the download button and click it
                wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="downloadScoreSiteButton"]')))
                download_button = driver.find_element(By.XPATH, '//*[@id="downloadScoreSiteButton"]')
                download_button.click()
                while (
                    len(
                        the_file_s := [
                            file
                            for file in os.listdir(os.path.expanduser(os.path.expandvars(download_path)))
                            if re.search(r"score-site-result-table\s*\(*[0-9]*\)*\.tsv", file)
                        ]
                    )
                    < 1
                ):
                    time.sleep(0.01)
                os.rename(
                    os.path.expanduser(os.path.expandvars(os.path.join(download_path, the_file_s[0]))),
                    new_name := os.path.expanduser(
                        os.path.expandvars(os.path.join(download_path, f"{which_short_site}-{p}`{i}.tsv"))
                    ),
                )
                while os.path.getsize(new_name) < 20_000:
                    time.sleep(0.01)

                # except next button not found error
            # Find the next pagination button and click it
            if p != max_page - 1:
                next_button = driver.find_element(By.LINK_TEXT, f"{p + 1}")
                actions = ActionChains(driver)
                actions.move_to_element(next_button).perform()
                wait.until(EC.element_to_be_clickable(next_button)).click()
    # Import NoSuchElementException from selenium.common.exceptions
    except Exception as e:
        informative_exception(e)

    # Close the browser


if __name__ == "__main__": # pragma: no cover
    main()
