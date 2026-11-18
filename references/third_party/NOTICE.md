# Third-party notices

AugMix augmentation operator semantics in `src/data.py` are adapted from
`google-research/augmix/augmentations.py`.

Original notice:

> Copyright 2019 Google LLC
>
> Licensed under the Apache License, Version 2.0 (the "License");
> you may not use this file except in compliance with the License.
> You may obtain a copy of the License at
>
> https://www.apache.org/licenses/LICENSE-2.0
>
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS,
> WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
> See the License for the specific language governing permissions and
> limitations under the License.

The accompanying `AugMix-LICENSE` contains the full license. The adaptation
organizes operators into a stateless per-image/epoch/view dataset pipeline,
uses Pillow's current transform/resampling enums, and changes surrounding
data-loading and random-number handling. This is not an unmodified copy of
the upstream training program. The frozen source is retained byte-for-byte;
this notice accompanies it without rewriting its recorded checksum.
