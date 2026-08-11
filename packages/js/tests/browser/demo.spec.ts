import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const DEMO_ORIGIN = "http://127.0.0.1:4173";
let offOriginRequests: string[] = [];
let browserErrors: string[] = [];

test.beforeEach(async ({ context, page }) => {
  offOriginRequests = [];
  browserErrors = [];

  await context.route("**/*", async (route) => {
    const requestUrl = new URL(route.request().url());
    if (
      (requestUrl.protocol === "http:" || requestUrl.protocol === "https:") &&
      requestUrl.origin !== DEMO_ORIGIN
    ) {
      offOriginRequests.push(requestUrl.href);
      await route.abort("blockedbyclient");
      return;
    }
    await route.continue();
  });
  page.on("console", (message) => {
    if (message.type() === "error") {
      browserErrors.push(message.text());
    }
  });
  page.on("pageerror", (error) => browserErrors.push(error.message));

  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "Experimental Attribution Viewer Demo",
  );

  test.info().annotations.push({
    type: "security-boundary",
    description: "Production demo must make no off-origin requests.",
  });
});

test.afterEach(() => {
  expect(offOriginRequests, "unexpected off-origin browser requests").toEqual(
    [],
  );
  expect(browserErrors, "browser console or page errors").toEqual([]);
});

test("updates synthetic attribution state through labelled controls", async ({
  page,
}) => {
  await expect(page.getByRole("note")).toContainText(
    "does not implement or execute any explainer",
  );

  await page.getByLabel("Task type").selectOption("Tabular Classification");
  await expect(page.getByLabel("Model label")).toHaveValue("XGBoost");
  await expect(page.getByLabel("Target class")).toHaveValue("Approved");
  await expect(page.getByRole("status")).toContainText(
    "Showing 4 synthetic attributions for Approved using the SHAP label",
  );
  await expect(
    page.getByLabel("credit_score: +0.6500 (positive)"),
  ).toBeVisible();

  await page.getByLabel("Target class").selectOption("Denied");
  await expect(
    page.getByLabel("credit_score: -0.7000 (negative)"),
  ).toBeVisible();

  await page.getByLabel("Feature name").fill("manual_review");
  await page.getByLabel("Attribution value").fill("0.33");
  await page.getByRole("button", { name: "Add" }).click();
  await expect(
    page.getByLabel("manual_review: +0.3300 (positive)"),
  ).toBeVisible();

  await page.getByRole("button", { name: "Remove manual_review" }).click();
  await expect(
    page.getByLabel("manual_review: +0.3300 (positive)"),
  ).toHaveCount(0);
});

test("supports keyboard traversal and exposes signed meter values", async ({
  page,
}) => {
  const taskSelect = page.getByLabel("Task type");
  const modelSelect = page.getByLabel("Model label");
  await taskSelect.focus();
  await page.keyboard.press("Tab");
  await expect(modelSelect).toBeFocused();

  const firstMeter = page.getByRole("meter").first();
  await expect(firstMeter).toHaveAttribute(
    "aria-valuetext",
    /\((positive|negative|zero)\)$/,
  );
  await expect(page.getByRole("main")).toContainText(
    "Top Feature Attributions",
  );

  await page.setViewportSize({ width: 320, height: 800 });
  const overflowsViewport = await page.evaluate(
    () =>
      document.documentElement.scrollWidth >
      document.documentElement.clientWidth,
  );
  expect(
    overflowsViewport,
    "demo must reflow without horizontal page scrolling at 320 CSS px",
  ).toBe(false);
});

test("has no detectable WCAG A or AA violations in the initial and empty states", async ({
  page,
}) => {
  const tags = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"];
  const initialResults = await new AxeBuilder({ page })
    .withTags(tags)
    .analyze();
  expect(initialResults.violations).toEqual([]);

  const removeButtons = page.getByRole("button", { name: /^Remove / });
  while ((await removeButtons.count()) > 0) {
    await removeButtons.first().click();
  }
  await expect(
    page.getByText("No features defined for this class yet."),
  ).toBeVisible();
  await expect(
    page.getByRole("status").filter({ hasText: "No feature attributions" }),
  ).toContainText("No feature attributions were supplied");

  const emptyResults = await new AxeBuilder({ page }).withTags(tags).analyze();
  expect(emptyResults.violations).toEqual([]);
});
