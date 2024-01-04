import requests
import xarray as xr
import io

class DataLoader:
    """
    Generic data loader class for fetching datasets via URLs.
    """
    
    def _load_via_url(self, urls):
        """
        Loads datasets from a list of URLs and concatenates them along the 'time' dimension.

        Args:
            urls (list): List of URLs pointing to dataset files.

        Returns:
            xr.Dataset: Xarray dataset containing concatenated data.
        """
        
        return xr.concat(
            [
                xr.open_dataset(io.BytesIO(requests.get(url, allow_redirects=True).content))
                for url in urls
            ],
            dim='time'
        )