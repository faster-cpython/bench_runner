const ORGANIZATION = "{{public_org}}";
const REPO = "{{public_repo}}";
const BRANCH = "main";
const REPO_URL = `https://raw.githubusercontent.com/${ORGANIZATION}/${REPO}/refs/heads/${BRANCH}/results/`;

let head_idx = null;
let base_idx = null;
let plot_diff = null;
let worker = null;
let python_ready = false;
let gridOptions = null;
let head_grid = null;
let base_grid = null;

function toggle_button(e, state) {
    if (state) {
        e.classList.remove("btn-outline-primary");
        e.classList.add("btn-primary");
    } else {
        e.classList.remove("btn-primary");
        e.classList.add("btn-outline-primary");
    }
}

function update_buttons() {
    let ready = (python_ready && head_idx !== null && base_idx !== null);
    document.querySelector("#go").disabled = !ready;
    toggle_button(document.querySelector("#go"), ready);
    document.querySelector("#merge_base").disabled = !(
        head_idx !== null &&
        gridOptions !== null &&
        gridOptions._bases[head_idx] !== null
    );
    toggle_button(document.querySelector("#head_ready"), head_idx !== null);
    toggle_button(document.querySelector("#base_ready"), base_idx !== null);
}

function setup_quick_access(bases) {
    let quick_select = document.querySelector("#quick_select");
    for (base of bases) {
        let button = document.createElement("button");
        button.type = "button";
        button.className = "btn btn-outline-primary";
        button.innerText = base;
        quick_select.appendChild(button);
        button.addEventListener("click", async (e) => {
            await base_grid.setColumnFilterModel("ref", {
                filterType: "text",
                type: "equals",
                filter: "v" + e.target.innerText,
            });
            await base_grid.onFilterChanged();
        });
    }

    document
        .querySelector("#merge_base")
        .addEventListener("click", async () => {
            let base = gridOptions._base_results[head_idx];
            base_grid.forEachNode((node) => {
                if (node.sourceRowIndex == base) {
                    node.setSelected(true);
                    base_grid.ensureNodeVisible(node);
                } else {
                    node.setSelected(false);
                }
            });
        });
}

async function setup_grid() {
    gridOptions = await fetch("index.json").then((response) => {
        return response.json();
    });

    setup_quick_access(gridOptions._bases);

    update_buttons();

    gridOptions.rowData = gridOptions.rowData.map((row) => {
        return {
            date: row[0],
            hash: row[1],
            fork: row[2],
            ref: row[3],
            version: row[4],
            runner: row[5],
            flags: row[6],
        }
    });

    gridOptions.columnDefs = [
        { field: "date", filter: "agDateColumnFilter" },
        { field: "hash" },
        { field: "fork" },
        { field: "ref" },
        { field: "version" },
        { field: "runner" },
        { field: "flags" },
    ];

    gridOptions.defaultColDef = {
        filter: "agTextColumnFilter",
        floatingFilter: true,
        sortable: true,
    };

    gridOptions.rowSelection = { mode: "singleRow" };

    gridOptions.onRowSelected = function (event) {
        let nodes = head_grid.getSelectedNodes();
        head_idx = (nodes.length == 1) ? nodes[0].sourceRowIndex : null;
        update_buttons();
    };

    gridOptions.autoSizeStrategy = { type: "SizeColumnsToContentStrategy" };

    const head_grid_element = document.querySelector("#head_grid");
    head_grid = agGrid.createGrid(head_grid_element, gridOptions);

    gridOptions.onRowSelected = function (event) {
        let nodes = base_grid.getSelectedNodes();
        base_idx = (nodes.length == 1) ? nodes[0].sourceRowIndex : null;
        update_buttons();
    };

    const base_grid_element = document.querySelector("#base_grid");
    base_grid = agGrid.createGrid(base_grid_element, gridOptions);
}

async function main() {
    document.querySelector("#go").addEventListener("click", async () => {
        let go_button = document.querySelector("#go");
        go_button.innerHTML = '<div class="spinner-border spinner-border-sm" role="status"/>';
        go_button.disabled = true;
        toggle_button(go_button, false);
        let head_url = REPO_URL + gridOptions._index[head_idx];
        let base_url = REPO_URL + gridOptions._index[base_idx];
        let [head_data, base_data] = await Promise.all([
            fetch(head_url).then((response) => {
                return response.text();
            }),
            fetch(base_url).then((response) => {
                return response.text();
            }),
        ]);

        worker.postMessage([base_url, base_data, head_url, head_data]);
    });

    worker = new Worker("worker.js", { type: "module" });
    worker.onmessage = (e) => {
        if (e.data == "READY") {
            let ready_button = document.querySelector("#ready");
            python_ready = true;
            ready_button.innerHTML = "Ready";
            toggle_button(ready_button, python_ready);
        } else {
            let go_button = document.querySelector("#go");
            go_button.innerHTML = "Go";
            document.querySelector("#plot").innerHTML = e.data;
        }
        update_buttons();
    };

    await setup_grid();
}
